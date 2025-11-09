
import torch
import yaml
import argparse
import os
import time
from prettytable import PrettyTable

# --- Import Project Modules ---
from common.problem import VRPProblem
from common.env import VRPEnvironment
from model.policy import PolicyNetwork

# --- Import Benchmark Solvers ---
from benchmarks.greedy_solver import solve_greedy
from benchmarks.ortools_solver import solve_or_tools

def evaluate_rl_model(problem, checkpoint_path, config):
    """
    Evaluates the trained Reinforcement Learning model.
    This function is a streamlined version of evaluate.py.
    """
    start_time = time.time()
    device = problem.device

    # --- Load Model ---
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    policy_net_params = config['model_params'].copy()
    policy_net_params.pop('num_customers', None)
    
    policy_net = PolicyNetwork(**policy_net_params).to(device)
    policy_net.load_state_dict(checkpoint['state_dict'])
    policy_net.eval()

    # --- Generate Solution ---
    env = VRPEnvironment(problem)
    problem_features, dynamic_features = env.reset()
    
    embed_dim = config['model_params']['embed_dim']
    num_vehicles = problem.num_vehicles
    local_h = torch.zeros((1, num_vehicles, embed_dim), device=device)
    global_h = torch.zeros((1, embed_dim), device=device)
    recorder_hidden_states = (local_h, global_h)

    with torch.no_grad():
        while not env.all_done():
            mask = env.get_mask().unsqueeze(0)
            log_probs, recorder_hidden_states = policy_net(
                problem_features, dynamic_features, recorder_hidden_states, mask
            )
            actions = torch.argmax(log_probs, dim=-1).squeeze(0)
            (problem_features, dynamic_features), _, _ = env.step(actions)

    total_cost = env.get_total_cost().item()
    duration = time.time() - start_time
    
    return total_cost, env.get_tours(), duration


def main():
    """
    Main function to run the benchmark comparison.
    """
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Benchmark RL model against Greedy and OR-Tools")
    parser.add_argument("checkpoint_path", type=str, help="Path to the RL model checkpoint file (.pt or .pth.tar)")
    parser.add_argument("--nodes", type=int, default=20, help="Number of customer nodes for the test problem.")
    parser.add_argument("--vehicles", type=int, default=4, help="Number of vehicles for the test problem.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generating the test problem.")
    args = parser.parse_args()

    # --- 2. Setup ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- 3. Generate a Single, Fixed Problem Instance ---
    print(f"Generating a benchmark problem with {args.nodes} nodes, {args.vehicles} vehicles, and seed {args.seed}...")
    problem = VRPProblem.generate_random_problem(
        num_nodes=args.nodes,
        num_vehicles=args.vehicles,
        device=device,
        seed=args.seed
    )
    print("Benchmark problem generated.\n")

    # --- 4. Run All Solvers ---
    results = {}

    # Run RL Model
    print("Running Reinforcement Learning model...")
    rl_cost, _, rl_duration = evaluate_rl_model(problem, args.checkpoint_path, config)
    results['RL Model'] = {'cost': rl_cost, 'duration': rl_duration}
    print("RL model finished.")

    # Run Greedy Solver
    print("Running Greedy (Nearest Neighbor) solver...")
    greedy_cost, _, greedy_duration = solve_greedy(problem)
    results['Greedy'] = {'cost': greedy_cost, 'duration': greedy_duration}
    print("Greedy solver finished.")

    # Run OR-Tools Solver
    print("Running Google OR-Tools solver...")
    or_cost, _, or_duration = solve_or_tools(problem)
    results['OR-Tools'] = {'cost': or_cost, 'duration': or_duration}
    print("OR-Tools solver finished.\n")

    # --- 5. Display Results ---
    table = PrettyTable()
    table.field_names = ["Algorithm", "Total Cost (lower is better)", "Time (seconds)"]
    table.align["Algorithm"] = "l"
    table.align["Total Cost (lower is better)"] = "r"
    table.align["Time (seconds)"] = "r"

    for name, res in results.items():
        table.add_row([name, f"{res['cost']:.4f}", f"{res['duration']:.4f}"])

    print("--- Benchmark Results ---")
    print(table)


if __name__ == "__main__":
    # First, check if the prettytable library is installed
    try:
        import prettytable
    except ImportError:
        print("Please install the 'prettytable' library to run this script:")
        print("pip install prettytable")
        exit(1)
        
    main()
