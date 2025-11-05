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
        The main trainer class.

        Args:
            config (dict): A dictionary containing training configuration.
        """
        self.config = config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy and baseline networks
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

        self.update_baseline() # Initialize baseline with policy weights

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['training_params']['learning_rate'])

        # Exponential moving average for baseline in the first epoch
        self.ema_baseline = torch.tensor(0.0)
        self.ema_alpha = 0.9
        self.baseline_rewards_for_ttest = []

    def train_batch(self, batch_problems):
        """
        Trains the model on a single batch of problems.
        """
        batch_size = len(batch_problems)

        # --- Rollout with Policy Network ---
        rewards, log_probs_all = self.rollout(self.policy_net, batch_problems, is_greedy=False)

        # --- Calculate Baseline ---
        with torch.no_grad():
            baseline_rewards, _ = self.rollout(self.baseline_net, batch_problems, is_greedy=True)

        # --- Calculate Loss and Update ---
        advantage = rewards - baseline_rewards

        # Sum log probabilities over the trajectory and vehicles
        total_log_probs = torch.sum(torch.cat(log_probs_all, dim=1), dim=(1, 2))

        # Policy gradient loss. We use a negative sign because optimizers minimize a loss,
        # but we want to maximize the objective function (reward).
        loss = -torch.mean(advantage * total_log_probs)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Gradient clipping
        self.optimizer.step()

        return rewards.mean().item(), loss.item()

    def rollout(self, model, problems, is_greedy):
        """
        Performs a rollout for a batch of problems using the given model.
        """
        batch_size = len(problems)
        envs = [Env(p) for p in problems]

        # Prepare initial inputs
        features = self._get_features_from_problems(problems).to(self.device)

        # Initialize hidden states for recorders
        local_h = torch.zeros(batch_size, self.config['model_params']['num_vehicles'], self.config['model_params']['embed_dim']).to(self.device)
        global_h = torch.zeros(batch_size, self.config['model_params']['embed_dim']).to(self.device)

        log_probs_list = []

        # Main decoding loop
        while not all(env.all_finished() for env in envs):
            vehicle_states = self._get_vehicle_states(envs).to(self.device)
            masks = self._get_masks(envs).to(self.device)

            log_probs, (local_h, global_h) = model(features, vehicle_states, (local_h, global_h), masks)

            # Select actions
            if is_greedy:
                actions = torch.argmax(log_probs, dim=-1)
            else:
                actions = Categorical(logits=log_probs).sample()

            # Store log probabilities of chosen actions
            chosen_log_probs = torch.gather(log_probs, 2, actions.unsqueeze(-1)).squeeze(-1)
            log_probs_list.append(chosen_log_probs.unsqueeze(1))

            # Step environments
            for i in range(batch_size):
                for v_idx in range(self.config['model_params']['num_vehicles']):
                    action = actions[i, v_idx].item()
                    envs[i].step(v_idx, action)

        # Calculate final rewards (negative costs)
        rewards = torch.tensor([-env.calculate_costs()[0] for env in envs], dtype=torch.float32).to(self.device)

        return rewards, log_probs_list

    def update_baseline(self):
        """Copies the policy network's weights to the baseline network."""
        self.baseline_net.load_state_dict(self.policy_net.state_dict())

    def check_baseline_update(self, epoch_rewards):
        """
        Performs a t-test to check if the baseline should be updated.
        """
        policy_rewards = np.array([r for r, _ in epoch_rewards])
        baseline_rewards = np.array([b for _, b in epoch_rewards])

        if len(policy_rewards) > 1:
            t_stat, p_value = ttest_rel(policy_rewards, baseline_rewards)
            # One-sided test: if policy is significantly better
            if t_stat > 0 and p_value / 2 < 0.05:
                print("--- Updating baseline network ---")
                self.update_baseline()

    def _get_features_from_problems(self, problems):
        """Prepares the feature tensor for a batch of problems."""
        # This needs to be adapted based on the final feature engineering
        # For now, a simplified version
        all_features = []
        for p in problems:
            # loc(2)+demand(1)+serv_time(1)+tw(2) = 6
            features = torch.cat([
                p.locations,
                p.demands.unsqueeze(1),
                p.service_times.unsqueeze(1),
                p.time_windows
            ], dim=1)
            all_features.append(features.unsqueeze(0))
        return torch.cat(all_features, dim=0)

    def _get_vehicle_states(self, envs):
        """Gathers current vehicle states from a batch of environments."""
        states = []
        for env in envs:
            # pos(2) + load(1)
            vehicle_locs = env.problem.locations[env.current_locations]
            loads = env.remaining_capacities.unsqueeze(1) / env.problem.capacities[0] # Normalize load
            state = torch.cat([vehicle_locs, loads], dim=1)
            states.append(state.unsqueeze(0))
        return torch.cat(states, dim=0)

    def _get_masks(self, envs):
        """Gathers masks from a batch of environments."""
        masks = []
        for env in envs:
            v_masks = []
            for v_idx in range(self.config['model_params']['num_vehicles']):
                v_masks.append(env.get_mask(v_idx).unsqueeze(0))
            masks.append(torch.cat(v_masks, dim=0).unsqueeze(0))
        return torch.cat(masks, dim=0)

if __name__ == '__main__':
    # Example usage
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

    # Generate a dummy batch of problems
    problems = [Problem.generate_random_instance(config['num_customers'], config['num_vehicles']) for _ in range(config['batch_size'])]

    print("Starting one batch training...")
    avg_reward, loss = trainer.train_batch(problems)
    print(f"Batch finished. Average Reward: {avg_reward:.3f}, Loss: {loss:.4f}")
    print("Training test passed.")

