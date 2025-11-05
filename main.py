import yaml
from training.trainer import Trainer
from common.problem import Problem

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Configuration loaded:")
    print(config)

    trainer = Trainer(config)

    num_epochs = config['training_params']['num_epochs']
    batches_per_epoch = config['training_params']['batches_per_epoch']
    batch_size = config['training_params']['batch_size']

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_losses = []

        for batch_idx in range(batches_per_epoch):
            # Generate a new batch of random problems for each step
            problems = [
                Problem.generate_random_instance(
                    config['model_params']['num_customers'],
                    config['model_params']['num_vehicles']
                ) for _ in range(batch_size)
            ]

            avg_reward, loss = trainer.train_batch(problems)

            epoch_rewards.append(avg_reward)
            epoch_losses.append(loss)

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{batches_per_epoch} | "
                      f"Avg Reward: {avg_reward:.3f} | Loss: {loss:.4f}")

        # End of epoch
        avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Average Reward: {avg_epoch_reward:.3f}")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        print("---------------------------\n")

        # Check if baseline needs to be updated (simplified check for now)
        # A proper implementation would collect policy and baseline rewards separately
        # trainer.check_baseline_update(...)

if __name__ == '__main__':
    main()

