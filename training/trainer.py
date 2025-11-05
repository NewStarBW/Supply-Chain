import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from scipy.stats import ttest_rel

from common.problem import Problem
from common.env import Env
from model.policy import PolicyNetwork

class Trainer:
    def __init__(self, config):
        """
        主训练器类。

        参数:
            config (dict): 包含训练配置的字典。
        """
        self.config = config

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 策略网络和基线网络
        self.policy_net = PolicyNetwork(
            input_dim=config['model_params']['input_dim'],
            embed_dim=config['model_params']['embed_dim'],
            num_heads=config['model_params']['num_heads'],
            num_layers=config['model_params']['num_layers'],
            num_vehicles=config['model_params']['num_vehicles']
        ).to(self.device)

        self.baseline_net = PolicyNetwork(
            input_dim=config['model_params']['input_dim'],
            embed_dim=config['model_params']['embed_dim'],
            num_heads=config['model_params']['num_heads'],
            num_layers=config['model_params']['num_layers'],
            num_vehicles=config['model_params']['num_vehicles']
        ).to(self.device)

        self.update_baseline() # 使用策略网络的权重初始化基线网络

        # 优化器和学习率调度器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['training_params']['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['scheduler_params']['lr_decay_step_size'],
            gamma=config['scheduler_params']['lr_decay_gamma']
        )

        # 用于检查点的状态
        self.start_epoch = 0
        self.ema_baseline = torch.tensor(0.0)

        # 用于t检验
        self.policy_rewards_for_ttest = []
        self.baseline_rewards_for_ttest = []

    def train_batch(self, batch_problems, current_epoch):
        """
        在单批次问题上训练模型。
        """
        batch_size = len(batch_problems)

        # --- 使用策略网络进行部署 ---
        rewards, log_probs_all = self.rollout(self.policy_net, batch_problems, is_greedy=False)

        # --- 计算基线 ---
        with torch.no_grad():
            if current_epoch == 0:
                # 第一轮：使用指数移动平均作为基线
                if self.ema_baseline.item() == 0.0: # 如果是第一个批次
                    self.ema_baseline = rewards.mean()
                else:
                    self.ema_baseline = self.ema_alpha * self.ema_baseline + (1 - self.ema_alpha) * rewards.mean()
                baseline_rewards = self.ema_baseline
            else:
                # 后续轮次：使用基线网络
                baseline_rewards, _ = self.rollout(self.baseline_net, batch_problems, is_greedy=True)

        # 存储用于t检验的奖励
        self.policy_rewards_for_ttest.extend(rewards.cpu().numpy())
        if current_epoch > 0:
            self.baseline_rewards_for_ttest.extend(baseline_rewards.cpu().numpy())

        # --- 计算损失并更新 ---
        advantage = rewards - baseline_rewards

        # 对轨迹和车辆的对数概率求和
        total_log_probs = torch.sum(torch.cat(log_probs_all, dim=1), dim=(1, 2))

        # 策略梯度损失。我们使用负号，因为优化器是最小化损失，
        # 但我们希望最大化目标函数（奖励）。
        loss = -torch.mean(advantage * total_log_probs)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # 梯度裁剪
        self.optimizer.step()

        return rewards.mean().item(), loss.item(), baseline_rewards.mean().item() if torch.is_tensor(baseline_rewards) else baseline_rewards

    def rollout(self, model, problems, is_greedy):
        """
        使用给定模型为一批问题执行一次完整的部署（rollout）。
        """
        batch_size = len(problems)
        envs = [Env(p) for p in problems]

        # 准备初始输入
        features = self._get_features_from_problems(problems).to(self.device)

        # 初始化记录器的隐藏状态
        local_h = torch.zeros(batch_size, self.config['model_params']['num_vehicles'], self.config['model_params']['embed_dim']).to(self.device)
        global_h = torch.zeros(batch_size, self.config['model_params']['embed_dim']).to(self.device)

        log_probs_list = []

        # 主解码循环
        while not all(env.all_finished() for env in envs):
            vehicle_states = self._get_vehicle_states(envs).to(self.device)
            masks = self._get_masks(envs).to(self.device)

            log_probs, (local_h, global_h) = model(features, vehicle_states, (local_h, global_h), masks)

            # 选择动作
            if is_greedy:
                actions = torch.argmax(log_probs, dim=-1)
            else:
                actions = Categorical(logits=log_probs).sample()

            # 存储被选中动作的对数概率
            chosen_log_probs = torch.gather(log_probs, 2, actions.unsqueeze(-1)).squeeze(-1)
            log_probs_list.append(chosen_log_probs.unsqueeze(1))

            # 在环境中执行步骤
            for i in range(batch_size):
                for v_idx in range(self.config['model_params']['num_vehicles']):
                    action = actions[i, v_idx].item()
                    envs[i].step(v_idx, action)

        # 计算最终奖励（成本的负数）
        rewards = torch.tensor([-env.calculate_costs()[0] for env in envs], dtype=torch.float32).to(self.device)

        return rewards, log_probs_list

    def update_baseline(self):
        """将策略网络的权重复制到基线网络。"""
        self.baseline_net.load_state_dict(self.policy_net.state_dict())

    def check_baseline_update(self, epoch_rewards):
        """
        执行t检验以检查是否应更新基线。
        """
        # 仅在有足够数据且不是第一轮时执行t检验
        if len(self.policy_rewards_for_ttest) > 1 and len(self.baseline_rewards_for_ttest) > 1:
            t_stat, p_value = ttest_rel(self.policy_rewards_for_ttest, self.baseline_rewards_for_ttest)
            # 单边检验：如果策略显著更优
            if t_stat > 0 and p_value / 2 < 0.05:
                print("--- 基线表现统计上更差。正在更新基线网络。 ---")
                self.update_baseline()

        # 为下一轮的数据清空列表
        self.policy_rewards_for_ttest.clear()
        self.baseline_rewards_for_ttest.clear()

    def save_checkpoint(self, epoch, filename="checkpoint.pth.tar"):
        """将训练器的状态保存到文件。"""
        state = {
            'epoch': epoch + 1,
            'state_dict': self.policy_net.state_dict(),
            'baseline_state_dict': self.baseline_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema_baseline': self.ema_baseline
        }
        torch.save(state, filename)
        print(f"检查点已于轮次 {epoch} 保存")

    def load_checkpoint(self, filename="checkpoint.pth.tar"):
        """从文件加载训练器的状态。"""
        try:
            state = torch.load(filename)
            self.start_epoch = state['epoch']
            self.policy_net.load_state_dict(state['state_dict'])
            self.baseline_net.load_state_dict(state['baseline_state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.ema_baseline = state['ema_baseline']
            print(f"检查点已加载。从轮次 {self.start_epoch} 继续。")
        except FileNotFoundError:
            print("未找到检查点。从头开始训练。")
        except Exception as e:
            print(f"无法加载检查点: {e}。从头开始训练。")

    def _get_features_from_problems(self, problems):
        """为一批问题准备特征张量。"""
        # 这里需要根据最终的特征工程进行调整
        # 目前是一个简化版本
        all_features = []
        for p in problems:
            # 位置(2)+需求(1)+服务时间(1)+时间窗(2) = 6
            features = torch.cat([
                p.locations,
                p.demands.unsqueeze(1),
                p.service_times.unsqueeze(1),
                p.time_windows
            ], dim=1)
            all_features.append(features.unsqueeze(0))
        return torch.cat(all_features, dim=0)

    def _get_vehicle_states(self, envs):
        """从一批环境中收集当前车辆状态。"""
        states = []
        for env in envs:
            # 位置(2) + 负载(1)
            vehicle_locs = env.problem.locations[env.current_locations]
            loads = env.remaining_capacities.unsqueeze(1) / env.problem.capacities[0] # 归一化负载
            state = torch.cat([vehicle_locs, loads], dim=1)
            states.append(state.unsqueeze(0))
        return torch.cat(states, dim=0)

    def _get_masks(self, envs):
        """从一批环境中收集掩码。"""
        masks = []
        for env in envs:
            v_masks = []
            for v_idx in range(self.config['model_params']['num_vehicles']):
                v_masks.append(env.get_mask(v_idx).unsqueeze(0))
            masks.append(torch.cat(v_masks, dim=0).unsqueeze(0))
        return torch.cat(masks, dim=0)

if __name__ == '__main__':
    # 示例用法
    config = {
        'input_dim': 6,
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 3,
        'num_vehicles': 2,
        'num_customers': 20,
        'learning_rate': 1e-4,
        'batch_size': 4
    }

    trainer = Trainer(config)

    # 生成一个虚拟的问题批次
    problems = [Problem.generate_random_instance(config['num_customers'], config['num_vehicles']) for _ in range(config['batch_size'])]

    print("开始单批次训练...")
    avg_reward, loss = trainer.train_batch(problems)
    print(f"批次完成。平均奖励: {avg_reward:.3f}, 损失: {loss:.4f}")
    print("训练测试通过。")
