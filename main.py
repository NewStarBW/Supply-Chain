import yaml
import os
from training.trainer import Trainer
from common.problem import Problem

def main():
    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("配置已加载:")
    print(config)

    trainer = Trainer(config)

    # 检查检查点文件
    checkpoint_dir = "training/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth.tar")
    trainer.load_checkpoint(checkpoint_path)

    num_epochs = config['training_params']['num_epochs']
    batches_per_epoch = config['training_params']['batches_per_epoch']
    batch_size = config['training_params']['batch_size']

    print(f"\n从第 {trainer.start_epoch} 轮开始训练...")

    try:
        for epoch in range(trainer.start_epoch, num_epochs):
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

                avg_reward, loss, baseline_reward = trainer.train_batch(problems, epoch)

                epoch_rewards.append(avg_reward)
                epoch_losses.append(loss)

                if (batch_idx + 1) % 10 == 0:
                    print(f"轮次 {epoch+1}/{num_epochs} | 批次 {batch_idx+1}/{batches_per_epoch} | "
                          f"平均奖励: {avg_reward:.3f} | 基线: {baseline_reward:.3f} | 损失: {loss:.4f}")

            # 轮次结束
            trainer.scheduler.step() # 学习率衰减
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\n--- 第 {epoch+1} 轮总结 ---")
            print(f"平均奖励: {avg_epoch_reward:.3f}")
            print(f"平均损失: {avg_epoch_loss:.4f}")
            print(f"当前学习率: {trainer.optimizer.param_groups[0]['lr']:.6f}")

            # 检查是否需要更新基线模型
            trainer.check_baseline_update(epoch)

            # 保存检查点
            trainer.save_checkpoint(epoch, checkpoint_path)
            print("---------------------------\n")

    except KeyboardInterrupt:
        print("\n训练被用户中断。正在保存检查点...")
        # `epoch` 变量是最后完成的轮次数
        trainer.save_checkpoint(epoch, checkpoint_path)
        print("检查点已保存。正在退出。")

if __name__ == '__main__':
    main()
