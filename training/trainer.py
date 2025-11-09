import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from scipy.stats import ttest_rel

from common.problem import Problem
from common.env import Env
from model.policy import PolicyNetwork

class Trainer:
    def __init__(self, config, device):
        """
        主训练器类。

        参数:
            config (dict): 包含训练配置的字典。
            device (torch.device): 训练设备 (CPU 或 CUDA)。
        """
        self.config = config
        self.device = device

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

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['training_params']['learning_rate'])

        # 用于检查点的状态
        self.start_epoch = 0
        self.ema_baseline = torch.tensor(0.0)
        self.ema_alpha = config['training_params'].get('ema_alpha', 0.95) # 从配置加载或使用默认值
        self.entropy_coefficient = config['training_params'].get('entropy_coefficient', 0.0)
        self.advantage_normalization = config['training_params'].get('advantage_normalization', True)
        # 迟到偏置与时间归一化尺度（用于车辆状态）
        self.decoder_lateness_bias = config['training_params'].get('decoder_lateness_bias', 0.0)
        self.time_norm_scale = config['training_params'].get('time_norm_scale', 10.0)
        # 可选：训练时对惩罚部分加权（只影响训练，不改变评估中的惩罚计算）
        self.use_penalty_weight = config['training_params'].get('use_penalty_weight', False)
        self.penalty_weight = float(config['training_params'].get('penalty_weight', 1.0))

        # 用于t检验
        self.policy_rewards_for_ttest = []
        self.baseline_rewards_for_ttest = []

    def train_batch(self, batch_problems, current_epoch):
        """在单批次问题上训练模型并返回指标。

        返回:
            avg_reward, loss, baseline_reward, avg_length, avg_penalty, avg_early_penalty, avg_late_penalty
        """
        batch_size = len(batch_problems)

        compute_entropy = self.entropy_coefficient > 0.0
        (
            rewards,
            log_probs_all,
            entropies_all,
            lengths,
            penalties,
            early_penalties,
            late_penalties,
        ) = self.rollout(
            self.policy_net,
            batch_problems,
            is_greedy=False,
            compute_entropy=compute_entropy,
            apply_penalty_weight=self.use_penalty_weight,
        )

        # --- 计算基线 ---
        with torch.no_grad():
            if current_epoch == 0:
                if self.ema_baseline.item() == 0.0:  # 第一批次初始化
                    self.ema_baseline = rewards.mean()
                else:
                    self.ema_baseline = self.ema_alpha * self.ema_baseline + (1 - self.ema_alpha) * rewards.mean()
                baseline_rewards = self.ema_baseline
            else:
                baseline_rewards, _, _, _, _, _, _ = self.rollout(
                    self.baseline_net,
                    batch_problems,
                    is_greedy=True,
                    compute_entropy=False,
                    apply_penalty_weight=self.use_penalty_weight,
                )

        # 存储用于t检验的奖励
        self.policy_rewards_for_ttest.extend(rewards.cpu().numpy())
        if current_epoch > 0:
            self.baseline_rewards_for_ttest.extend(baseline_rewards.cpu().numpy())

        # --- 损失计算 ---
        advantage = rewards - baseline_rewards
        if self.advantage_normalization:
            adv_mean = advantage.mean()
            adv_std = advantage.std().clamp_min(1e-8)
            advantage = (advantage - adv_mean) / adv_std

        total_log_probs = torch.sum(torch.cat(log_probs_all, dim=1), dim=(1, 2))
        policy_loss = -torch.mean(advantage * total_log_probs)
        if self.entropy_coefficient > 0.0 and len(entropies_all) > 0:
            total_entropy = torch.sum(torch.cat(entropies_all, dim=1), dim=(1, 2))
            entropy_loss = -self.entropy_coefficient * torch.mean(total_entropy)
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)
        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        avg_len = lengths.mean().item()
        avg_pen = penalties.mean().item()
        avg_early = early_penalties.mean().item()
        avg_late = late_penalties.mean().item()

        return (
            rewards.mean().item(),
            loss.item(),
            baseline_rewards.mean().item() if torch.is_tensor(baseline_rewards) else baseline_rewards,
            avg_len,
            avg_pen,
            avg_early,
            avg_late,
        )

    def rollout(self, model, problems, is_greedy, return_tours=False, compute_entropy=True, apply_penalty_weight=False):
        """执行一次完整的顺序解码并返回轨迹及成本构成。"""
        batch_size = len(problems)
        envs = [Env(p) for p in problems]
        num_vehicles = self.config['model_params']['num_vehicles']

        # 准备初始输入
        features = self._get_features_from_problems(problems).to(self.device)
        node_embeddings, graph_embedding = model.encoder(features)

        # 初始化记录器的隐藏状态
        local_h = torch.zeros(batch_size, num_vehicles, self.config['model_params']['embed_dim']).to(self.device)
        global_h = torch.zeros(batch_size, self.config['model_params']['embed_dim']).to(self.device)

        log_probs_list = []
        entropies_list = [] # 存储每个步骤的熵

        # 主解码循环
        while not all(env.all_finished() for env in envs):
            step_log_probs = []
            step_actions = []
            step_entropies = [] if compute_entropy else None

            # --- 顺序解码循环 ---
            for v_idx in range(num_vehicles):
                vehicle_states = self._get_vehicle_states(envs).to(self.device)
                masks = self._get_masks(envs).to(self.device)

                # 更新记录器
                next_local_h_list = []
                for i in range(num_vehicles):
                    h_prev = local_h[:, i, :]
                    vehicle_state = vehicle_states[:, i, :]
                    h_next = model.local_recorders[i](vehicle_state, h_prev)
                    next_local_h_list.append(h_next.unsqueeze(1))
                next_local_h = torch.cat(next_local_h_list, dim=1)

                all_vehicle_states = vehicle_states.view(batch_size, -1)
                next_global_h = model.global_recorder(all_vehicle_states, global_h)

                # 为当前车辆构建观测
                observation = graph_embedding + next_local_h[:, v_idx, :] + next_global_h
                vehicle_mask = masks[:, v_idx, :]

                # 解码
                log_probs = model.decoder(observation, node_embeddings, vehicle_mask)
                # 迟到偏置：根据预测到达时间对 logits 做惩罚，帮助优先满足时间窗
                if self.decoder_lateness_bias and self.decoder_lateness_bias > 0.0:
                    # 特征中: loc(0:2), demand(2), service_time(3), time_windows(4:6)
                    locs = features[:, :, 0:2]
                    batch_idx = torch.arange(batch_size, device=self.device)
                    prev_idx = torch.tensor([env.current_locations[v_idx].item() for env in envs], device=self.device, dtype=torch.long)
                    prev_locs = locs[batch_idx, prev_idx]  # (B,2)
                    dists = torch.norm(locs - prev_locs.unsqueeze(1), dim=2)  # (B,N)
                    service_prev = features[batch_idx, prev_idx, 3]  # (B,)
                    current_times = torch.tensor([env.current_times[v_idx].item() for env in envs], device=self.device)
                    arrival = current_times.unsqueeze(1) + service_prev.unsqueeze(1) + dists  # 假设速度=1
                    l_j = features[:, :, 5]
                    lateness = torch.clamp(arrival - l_j, min=0.0)
                    # 返回仓库节点不计罚
                    lateness[:, 0] = 0.0
                    log_probs = log_probs - self.decoder_lateness_bias * lateness
                
                # 创建概率分布
                dist = Categorical(logits=log_probs)

                # 选择动作
                if is_greedy:
                    action = torch.argmax(log_probs, dim=-1)
                else:
                    action = dist.sample()

                # 存储动作、对数概率和熵
                step_actions.append(action)
                step_log_probs.append(dist.log_prob(action))
                if compute_entropy:
                    step_entropies.append(dist.entropy())

                # 在环境中立即执行步骤以更新掩码
                for i in range(batch_size):
                    envs[i].step(v_idx, action[i].item())
            
            # 更新隐藏状态
            local_h, global_h = next_local_h, next_global_h

            # 收集一个完整解码步骤（所有车辆）的对数概率和熵
            log_probs_list.append(torch.stack(step_log_probs, dim=1).unsqueeze(1))
            if compute_entropy:
                entropies_list.append(torch.stack(step_entropies, dim=1).unsqueeze(1))

        # 计算最终奖励及构成（环境返回原始长度与惩罚）
        breakdowns = [env.calculate_costs() for env in envs]
        lengths = torch.tensor([b[1] for b in breakdowns], dtype=torch.float32, device=self.device)
        penalties = torch.tensor([b[2] for b in breakdowns], dtype=torch.float32, device=self.device)
        early_penalties = torch.tensor([b[3] for b in breakdowns], dtype=torch.float32, device=self.device)
        late_penalties = torch.tensor([b[4] for b in breakdowns], dtype=torch.float32, device=self.device)

        # 用于训练的总成本：length + (penalty_weight * penalty) 当启用时
        if apply_penalty_weight:
            total_costs = lengths + self.penalty_weight * penalties
        else:
            total_costs = lengths + penalties

        rewards = -total_costs

        if return_tours:
            tours = []
            for env in envs:
                env_tours = []
                for route in env.routes:
                    route_list = list(route)
                    if route_list and route_list[-1] != 0:
                        route_list = route_list + [0]
                    env_tours.append(route_list)
                tours.append(env_tours)
            return (
                rewards,
                log_probs_list,
                entropies_list,
                lengths,
                penalties,
                early_penalties,
                late_penalties,
                tours,
            )

        return (
            rewards,
            log_probs_list,
            entropies_list,
            lengths,
            penalties,
            early_penalties,
            late_penalties,
        )

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

    def save_checkpoint(self, epoch, filename="checkpoint.pth.tar", history=None):
        """将训练器的状态和历史记录保存到文件。"""
        state = {
            'epoch': epoch + 1,
            'state_dict': self.policy_net.state_dict(),
            'baseline_state_dict': self.baseline_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema_baseline': self.ema_baseline,
            'history': history
        }
        torch.save(state, filename)
        print(f"检查点已于轮次 {epoch} 保存")

    def load_checkpoint(self, filename="checkpoint.pth.tar"):
        """从文件加载训练器的状态和历史记录。"""
        history = None
        try:
            state = torch.load(filename, map_location=self.device)
            self.start_epoch = state['epoch']
            self.policy_net.load_state_dict(state['state_dict'])
            self.baseline_net.load_state_dict(state['baseline_state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.ema_baseline = state['ema_baseline']
            history = state.get('history', None) # 加载历史记录
            print(f"检查点已加载。从轮次 {self.start_epoch} 继续。")
        except FileNotFoundError:
            print("未找到检查点。从头开始训练。")
        except Exception as e:
            print(f"无法加载检查点: {e}。从头开始训练。")
        return history

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
            loads = env.remaining_capacities.unsqueeze(1) / env.problem.capacities[0]  # 归一化负载
            times = (env.current_times.unsqueeze(1) / self.time_norm_scale)  # 归一化当前时间
            state = torch.cat([vehicle_locs, loads, times], dim=1)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(config, device)

    # 生成一个虚拟的问题批次
    problems = [Problem.generate_random_instance(config['num_customers'], config['num_vehicles']) for _ in range(config['batch_size'])]

    print("开始单批次训练...")
    avg_reward, loss, _ = trainer.train_batch(problems, current_epoch=0)
    print(f"批次完成。平均奖励: {avg_reward:.3f}, 损失: {loss:.4f}")
    print("训练测试通过。")
