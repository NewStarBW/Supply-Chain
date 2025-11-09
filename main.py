import yaml
import os
import torch
from training.trainer import Trainer
from common.problem import Problem
from common.visualizer import TrainingVisualizer

def main():
    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- 组合最终的模型和训练配置 ---
    active_size_name = config['training_params']['active_size']
    size_config = config['problem_presets'][active_size_name]
    
    # 将基础参数和规模参数合并到 model_params
    model_params = {**config['model_base_params'], **size_config}
    config['model_params'] = model_params # 更新全局config，以兼容旧代码

    # --- 打印关键训练参数 ---
    print("\n" + "="*50)

    print(f"当前训练规模: '{active_size_name}'")
    print(f"  - 客户数量: {model_params['num_customers']}")
    print(f"  - 车辆数量: {model_params['num_vehicles']}")
    print(f"  - 学习率: {config['training_params']['learning_rate']}")
    print(f"  - 总轮次: {config['training_params']['num_epochs']}")
    print(f"  - 每轮批次数: {config['training_params']['batches_per_epoch']}")
    print(f"  - 批次大小: {config['training_params']['batch_size']}")
    print("="*50 + "\n")

    # --- 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    trainer = Trainer(config, device)
    visualizer = TrainingVisualizer()

    # 根据问题规模动态设置检查点目录
    checkpoint_dir = os.path.join("training", "checkpoints", model_params['name'])
    print(f"模型将保存至: {checkpoint_dir}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth.tar")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pth.tar")
    
    # 加载检查点和历史数据
    history = trainer.load_checkpoint(checkpoint_path)
    if history:
        visualizer.rewards = history.get('rewards', [])
        visualizer.losses = history.get('losses', [])
        visualizer.learning_rates = history.get('lrs', [])
        print(f"已从检查点加载 {len(visualizer.rewards)} 轮的历史数据。")


    num_epochs = config['training_params']['num_epochs']
    batches_per_epoch = config['training_params']['batches_per_epoch']
    batch_size = config['training_params']['batch_size']
    # penalty weight schedule 配置（可选）
    pw_schedule = config['training_params'].get('penalty_weight_schedule', {})
    pw_enabled = pw_schedule.get('enabled', False)
    pw_start_ep = pw_schedule.get('start_epoch', 0)
    pw_end_ep = pw_schedule.get('end_epoch', num_epochs)
    pw_start_w = float(pw_schedule.get('start_weight', trainer.penalty_weight))
    pw_end_w = float(pw_schedule.get('end_weight', trainer.penalty_weight))
    
    # 提前停止参数
    early_stopping_config = config.get('early_stopping', {})
    patience = early_stopping_config.get('patience', 5)
    min_delta = early_stopping_config.get('min_delta', 0.01)
    best_reward = -float('inf')
    epochs_no_improve = 0
    if visualizer.rewards:
        best_reward = max(visualizer.rewards)


    print(f"\n从第 {trainer.start_epoch} 轮开始训练...")

    try:
        for epoch in range(trainer.start_epoch, num_epochs):
            # 每轮开始时更新训练器的 penalty_weight（如果启用了调度）
            if pw_enabled:
                if epoch <= pw_start_ep:
                    trainer.penalty_weight = pw_start_w
                elif epoch >= pw_end_ep:
                    trainer.penalty_weight = pw_end_w
                else:
                    frac = (epoch - pw_start_ep) / max(1, (pw_end_ep - pw_start_ep))
                    trainer.penalty_weight = pw_start_w + frac * (pw_end_w - pw_start_w)
                trainer.use_penalty_weight = True
            # 打印当前 penalty_weight 以便追踪
            if trainer.use_penalty_weight:
                print(f"当前训练 penalty_weight = {trainer.penalty_weight:.4f}")
            epoch_rewards = []
            epoch_losses = []

            for batch_idx in range(batches_per_epoch):
                # 为每一步生成一批新的随机问题
                problems = [
                    Problem.generate_random_instance(
                        config['model_params']['num_customers'],
                        config['model_params']['num_vehicles']
                    ) for _ in range(batch_size)
                ]

                (
                    avg_reward,
                    loss,
                    baseline_reward,
                    avg_len,
                    avg_pen,
                    avg_early,
                    avg_late,
                ) = trainer.train_batch(problems, epoch)

                epoch_rewards.append(avg_reward)
                epoch_losses.append(loss)
                # 可选：在后续扩展为可视化
                # 这里直接在输出中展示当批次的长度/惩罚构成

                # 调整打印频率，使其更适应不同的 batches_per_epoch
                print_interval = max(1, batches_per_epoch // 10) # 每轮打印10次
                if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == batches_per_epoch:
                    print(f"\r轮次 {epoch+1}/{num_epochs} | 批次 {batch_idx+1}/{batches_per_epoch} | "
                          f"平均奖励: {avg_reward:.3f} | 基线: {baseline_reward:.3f} | 损失: {loss:.4f} | "
                          f"长度: {avg_len:.2f} | 惩罚: {avg_pen:.2f} (早: {avg_early:.2f}, 晚: {avg_late:.2f})",
                          end="")

            # 轮次结束，换行
            print()
            
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            current_lr = trainer.optimizer.param_groups[0]['lr']
            
            visualizer.add_epoch_data(avg_epoch_reward, avg_epoch_loss, current_lr)
            visualizer.plot(epoch + 1)

            print(f"\n--- 第 {epoch+1} 轮总结 ---")
            print(f"平均奖励: {avg_epoch_reward:.3f}")
            print(f"平均损失: {avg_epoch_loss:.4f}")
            print(f"当前学习率: {current_lr:.6f}")

            # 检查是否需要更新基线模型
            trainer.check_baseline_update(epoch)

            # 保存检查点，包含历史数据（最新）
            history_to_save = {
                'rewards': visualizer.rewards,
                'losses': visualizer.losses,
                'lrs': visualizer.learning_rates
            }
            trainer.save_checkpoint(epoch, checkpoint_path, history_to_save)
            print("---------------------------\n")

            # 最优模型判定与保存
            improved = False
            if early_stopping_config.get('enabled', False):
                if avg_epoch_reward > best_reward + min_delta:
                    best_reward = avg_epoch_reward
                    epochs_no_improve = 0
                    improved = True
                    print(f"发现新的最佳奖励: {best_reward:.3f}。重置耐心计数器。")
                else:
                    epochs_no_improve += 1
                    print(f"奖励没有显著改善。耐心计数器: {epochs_no_improve}/{patience}。")

                if epochs_no_improve >= patience:
                    print(f"\n奖励连续 {patience} 轮没有显著改善。提前停止训练。")
                    # 在提前停止前，确保已保存当前最佳（若尚未保存则不影响）
                    break
            else:
                # 未启用提前停止时，严格使用“更大即更优”的规则
                if avg_epoch_reward > best_reward:
                    best_reward = avg_epoch_reward
                    improved = True

            if improved:
                trainer.save_checkpoint(epoch, best_checkpoint_path, history_to_save)
                print(f"最佳检查点已更新: {best_checkpoint_path}")

    except KeyboardInterrupt:
        print("\n训练被用户中断。正在保存检查点...")
        history_to_save = {
            'rewards': visualizer.rewards,
            'losses': visualizer.losses,
            'lrs': visualizer.learning_rates
        }
        trainer.save_checkpoint(epoch, checkpoint_path, history_to_save)
        print("检查点已保存。正在退出。")
    finally:
        visualizer.close()

if __name__ == '__main__':
    main()
