
"""Evaluate the trained policy network against baseline solvers."""
import argparse
import os
import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
# 使用无交互后端，确保在无显示环境下也能保存图片
plt.switch_backend("Agg")
import numpy as np
from torch.distributions import Categorical
import torch
import yaml

from benchmarks.greedy_solver import solve_greedy

try:
    from benchmarks.ortools_solver import solve_or_tools
    _HAS_OR_TOOLS = True
except ImportError:
    _HAS_OR_TOOLS = False

from common.problem import Problem
from common.env import Env
from training.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_problem(problem: Problem) -> Problem:
    problem.device = torch.device("cpu")
    problem.num_nodes = problem.num_customers
    return problem


def compute_cost_with_time_windows(problem: Problem, tours: List[List[int]]) -> Tuple[float, float, float]:
    """Compute cost with time-window penalties to match the training environment."""
    alpha = 0.5
    beta = 2.0

    locations = problem.locations
    service_times = problem.service_times
    time_windows = problem.time_windows
    speeds = problem.speeds

    total_length = 0.0
    total_penalty = 0.0

    for vehicle_idx, tour in enumerate(tours):
        if not tour:
            continue

        prev = 0
        current_time = 0.0
        speed = speeds[vehicle_idx].item() if speeds.numel() > 1 else speeds[0].item()

        for node in tour:
            dist = torch.norm(locations[prev] - locations[node], p=2).item()
            travel_time = dist / speed
            arrival_time = current_time + service_times[prev].item() + travel_time

            e_j, l_j = time_windows[node]
            early = max(0.0, e_j.item() - arrival_time)
            late = max(0.0, arrival_time - l_j.item())

            total_length += dist
            total_penalty += alpha * early + beta * late

            current_time = arrival_time
            prev = node

        # return to depot
        dist_back = torch.norm(locations[prev] - locations[0], p=2).item()
        total_length += dist_back

    return total_length + total_penalty, total_length, total_penalty


def plot_vrp_solution(problem: Problem, tours: List[List[int]], title: str, save_path: str) -> None:
    """Plot a VRP solution and save it to disk."""
    locations = problem.locations.cpu().numpy()
    depot = locations[0]
    customers = locations[1:] if locations.shape[0] > 1 else np.empty((0, 2))

    plt.figure(figsize=(6, 6))
    if customers.size > 0:
        plt.scatter(customers[:, 0], customers[:, 1], c="tab:blue", s=30, label="Customers")
    plt.scatter([depot[0]], [depot[1]], c="tab:red", marker="s", s=80, label="Depot")

    color_map = plt.get_cmap("tab10", max(1, len(tours)))
    for idx, route in enumerate(tours):
        # 绘制包含仓库起终点的完整路径
        full_route = [0] + list(route) + [0]
        coords = locations[np.array(full_route, dtype=int)] if len(full_route) > 1 else None
        if coords is None or coords.shape[0] < 2:
            continue
        plt.plot(coords[:, 0], coords[:, 1], color=color_map(idx), linewidth=1.5, label=f"Vehicle {idx + 1}")
        # 绘制该车访问的客户点（不含仓库）
        if len(route) > 0:
            cust_coords = locations[np.array(route, dtype=int)]
            plt.scatter(cust_coords[:, 0], cust_coords[:, 1], color=color_map(idx), s=20)

    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def generate_problems(num_instances: int, num_customers: int, num_vehicles: int, seed: int) -> List[Problem]:
    problems: List[Problem] = []
    for idx in range(num_instances):
        set_seed(seed + idx)
        problem = Problem.generate_random_instance(num_customers, num_vehicles)
        problems.append(prepare_problem(problem))
    return problems


def _solve_policy_greedy_single(trainer: Trainer, problem: Problem, collect_tours: bool) -> Tuple[float, List[List[int]], float, float, float]:
    """Run greedy decoding with the policy network on a single problem, return (cost, tours, time)."""
    model = trainer.policy_net
    device = trainer.device
    env = Env(problem)

    # Build static features once (same as trainer._get_features_from_problems for one instance)
    features = torch.cat([
        problem.locations,
        problem.demands.unsqueeze(1),
        problem.service_times.unsqueeze(1),
        problem.time_windows
    ], dim=1).unsqueeze(0).to(device)  # shape: (1, num_nodes, 6)

    with torch.no_grad():
        node_embeddings, graph_embedding = model.encoder(features)

    num_vehicles = problem.num_vehicles
    embed_dim = trainer.config['model_params']['embed_dim']
    batch_size = 1
    gamma = trainer.config['training_params'].get('decoder_lateness_bias', 0.0)
    time_scale = trainer.config['training_params'].get('time_norm_scale', 10.0)

    local_h = torch.zeros(batch_size, num_vehicles, embed_dim, device=device)
    global_h = torch.zeros(batch_size, embed_dim, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        while not env.all_finished():
            for v_idx in range(num_vehicles):
                # Build vehicle states (pos2 + load1)
                vehicle_locs = env.problem.locations[env.current_locations]  # (V, 2)
                loads = (env.remaining_capacities.unsqueeze(1) / env.problem.capacities[0])  # (V,1)
                times = (env.current_times.unsqueeze(1) / time_scale)  # (V,1)
                vehicle_states = torch.cat([vehicle_locs, loads, times], dim=1).unsqueeze(0).to(device)  # (1,V,4)

                # Update recorders for all vehicles
                next_local_h_list = []
                for i in range(num_vehicles):
                    h_prev = local_h[:, i, :]
                    vehicle_state_i = vehicle_states[:, i, :]
                    h_next = model.local_recorders[i](vehicle_state_i, h_prev)
                    next_local_h_list.append(h_next.unsqueeze(1))
                next_local_h = torch.cat(next_local_h_list, dim=1)

                all_vehicle_states = vehicle_states.view(batch_size, -1)
                next_global_h = model.global_recorder(all_vehicle_states, global_h)

                # Mask and decode for current vehicle
                vehicle_mask = env.get_mask(v_idx).unsqueeze(0).to(device)
                observation = graph_embedding + next_local_h[:, v_idx, :] + next_global_h
                logits = model.decoder(observation, node_embeddings, vehicle_mask)
                # Lateness shaping on logits (optional)
                if gamma and gamma > 0.0:
                    locs = features[:, :, 0:2]
                    # prev idx for this vehicle
                    prev_idx = torch.tensor([env.current_locations[v_idx].item()], device=device, dtype=torch.long)
                    prev_locs = locs[0, prev_idx]  # (1,2)
                    dists = torch.norm(locs - prev_locs.unsqueeze(1), dim=2)  # (1,N)
                    service_prev = features[0, prev_idx, 3]  # (1,)
                    current_time = torch.tensor([env.current_times[v_idx].item()], device=device)
                    arrival = current_time.unsqueeze(1) + service_prev.unsqueeze(1) + dists  # (1,N)
                    l_j = features[:, :, 5]
                    lateness = torch.clamp(arrival - l_j, min=0.0)
                    lateness[:, 0] = 0.0
                    logits = logits - gamma * lateness
                action = torch.argmax(logits, dim=-1).item()

                env.step(v_idx, action)

                # Commit hidden states
                local_h, global_h = next_local_h, next_global_h

    duration = time.perf_counter() - start

    # Extract tours as customer-only sequences (exclude depot nodes)
    tours = [[n for n in route if n != 0] for route in env.routes]
    cost, length, penalty = compute_cost_with_time_windows(problem, tours)
    return cost, tours if collect_tours else [], duration, length, penalty


def _solve_policy_sampled_single(trainer: Trainer, problem: Problem, samples: int, temperature: float, collect_tours: bool) -> Tuple[float, List[List[int]], float, float, float]:
    """Run sampled decoding K times and take the best (lowest cost). Return (cost, tours, time, length, penalty)."""
    model = trainer.policy_net
    device = trainer.device
    best = (float('inf'), [], 0.0, 0.0, 0.0)

    start_total = time.perf_counter()
    for _ in range(max(1, samples)):
        env = Env(problem)

        features = torch.cat([
            problem.locations,
            problem.demands.unsqueeze(1),
            problem.service_times.unsqueeze(1),
            problem.time_windows
        ], dim=1).unsqueeze(0).to(device)

        with torch.no_grad():
            node_embeddings, graph_embedding = model.encoder(features)

        num_vehicles = problem.num_vehicles
        embed_dim = trainer.config['model_params']['embed_dim']
        batch_size = 1
        gamma = trainer.config['training_params'].get('decoder_lateness_bias', 0.0)
        time_scale = trainer.config['training_params'].get('time_norm_scale', 10.0)

        local_h = torch.zeros(batch_size, num_vehicles, embed_dim, device=device)
        global_h = torch.zeros(batch_size, embed_dim, device=device)

        while not env.all_finished():
            for v_idx in range(num_vehicles):
                vehicle_locs = env.problem.locations[env.current_locations]
                loads = (env.remaining_capacities.unsqueeze(1) / env.problem.capacities[0])
                times = (env.current_times.unsqueeze(1) / time_scale)
                vehicle_states = torch.cat([vehicle_locs, loads, times], dim=1).unsqueeze(0).to(device)

                next_local_h_list = []
                for i in range(num_vehicles):
                    h_prev = local_h[:, i, :]
                    vehicle_state_i = vehicle_states[:, i, :]
                    h_next = model.local_recorders[i](vehicle_state_i, h_prev)
                    next_local_h_list.append(h_next.unsqueeze(1))
                next_local_h = torch.cat(next_local_h_list, dim=1)

                all_vehicle_states = vehicle_states.view(batch_size, -1)
                next_global_h = model.global_recorder(all_vehicle_states, global_h)

                vehicle_mask = env.get_mask(v_idx).unsqueeze(0).to(device)
                observation = graph_embedding + next_local_h[:, v_idx, :] + next_global_h
                logits = model.decoder(observation, node_embeddings, vehicle_mask)
                if gamma and gamma > 0.0:
                    locs = features[:, :, 0:2]
                    prev_idx = torch.tensor([env.current_locations[v_idx].item()], device=device, dtype=torch.long)
                    prev_locs = locs[0, prev_idx]
                    dists = torch.norm(locs - prev_locs.unsqueeze(1), dim=2)
                    service_prev = features[0, prev_idx, 3]
                    current_time = torch.tensor([env.current_times[v_idx].item()], device=device)
                    arrival = current_time.unsqueeze(1) + service_prev.unsqueeze(1) + dists
                    l_j = features[:, :, 5]
                    lateness = torch.clamp(arrival - l_j, min=0.0)
                    lateness[:, 0] = 0.0
                    logits = logits - gamma * lateness
                if temperature != 1.0:
                    logits = logits / float(temperature)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                env.step(v_idx, action)
                local_h, global_h = next_local_h, next_global_h

        tours = [[n for n in route if n != 0] for route in env.routes]
        cost, length, penalty = compute_cost_with_time_windows(problem, tours)
        if cost < best[0]:
            best = (cost, tours if collect_tours else [], 0.0, length, penalty)

    duration = time.perf_counter() - start_total
    # Attach measured duration
    return best[0], best[1], duration, best[3], best[4]


def evaluate_policy(trainer: Trainer, problems: List[Problem], collect_details: bool, decode: str = "greedy", samples: int = 1, temperature: float = 1.0) -> Dict[str, object]:
    trainer.policy_net.eval()
    costs: List[float] = []
    durations: List[float] = []
    lengths: List[float] = []
    penalties: List[float] = []
    tours: List[List[List[int]]] = []

    for problem in problems:
        if decode == "sample":
            cost, t, dur, length, penalty = _solve_policy_sampled_single(trainer, problem, samples, temperature, collect_details)
        else:
            cost, t, dur, length, penalty = _solve_policy_greedy_single(trainer, problem, collect_details)
        costs.append(cost)
        durations.append(dur)
        lengths.append(length)
        penalties.append(penalty)
        if collect_details:
            tours.append(t)

    result: Dict[str, object] = {
        "costs": np.array(costs),
        "times": np.array(durations),
        "lengths": np.array(lengths),
        "penalties": np.array(penalties),
    }
    if collect_details:
        result["tours"] = tours
    return result


def evaluate_baselines(problems: List[Problem], use_greedy: bool, use_or_tools: bool, collect_details: bool) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}

    if use_greedy:
        greedy_costs: List[float] = []
        greedy_times: List[float] = []
        greedy_tours: List[List[List[int]]] = []
        greedy_lengths: List[float] = []
        greedy_penalties: List[float] = []
        for problem in problems:
            _, tours, duration = solve_greedy(problem)
            cost_with_penalty, length, penalty = compute_cost_with_time_windows(problem, tours)
            greedy_costs.append(cost_with_penalty)
            greedy_times.append(duration)
            greedy_lengths.append(length)
            greedy_penalties.append(penalty)
            if collect_details:
                greedy_tours.append(tours)
        entry: Dict[str, object] = {
            "costs": np.array(greedy_costs),
            "times": np.array(greedy_times),
            "lengths": np.array(greedy_lengths),
            "penalties": np.array(greedy_penalties),
        }
        if collect_details:
            entry["tours"] = greedy_tours
        results["greedy"] = entry

    if use_or_tools and _HAS_OR_TOOLS:
        ortools_costs: List[float] = []
        ortools_times: List[float] = []
        ortools_tours: List[List[List[int]]] = []
        ortools_lengths: List[float] = []
        ortools_penalties: List[float] = []
        for problem in problems:
            _, tours, duration = solve_or_tools(problem)
            cost_with_penalty, length, penalty = compute_cost_with_time_windows(problem, tours)
            ortools_costs.append(cost_with_penalty)
            ortools_times.append(duration)
            ortools_lengths.append(length)
            ortools_penalties.append(penalty)
            if collect_details:
                ortools_tours.append(tours)
        entry = {
            "costs": np.array(ortools_costs),
            "times": np.array(ortools_times),
            "lengths": np.array(ortools_lengths),
            "penalties": np.array(ortools_penalties),
        }
        if collect_details:
            entry["tours"] = ortools_tours
        results["ortools"] = entry

    return results


def _print_results_table(policy_result: Dict[str, object], baselines: Dict[str, Dict[str, object]]) -> None:
    rows = []
    def add_row(name: str, costs: np.ndarray, times: np.ndarray, lengths: np.ndarray, penalties: np.ndarray):
        rows.append((name, costs.mean(), costs.std(), times.mean(), lengths.mean(), penalties.mean()))

    add_row("Policy", policy_result["costs"], policy_result["times"], policy_result.get("lengths", np.array([])), policy_result.get("penalties", np.array([])))
    if "greedy" in baselines:
        add_row("Greedy", baselines["greedy"]["costs"], baselines["greedy"]["times"], baselines["greedy"].get("lengths", np.array([])), baselines["greedy"].get("penalties", np.array([])))
    if "ortools" in baselines:
        add_row("OR-Tools", baselines["ortools"]["costs"], baselines["ortools"]["times"], baselines["ortools"].get("lengths", np.array([])), baselines["ortools"].get("penalties", np.array([])))

    # Print formatted table
    header = ("Method", "Avg Cost", "Std", "Avg Time (s)", "Avg Len", "Avg Pen")
    col_widths = [max(len(header[i]), max(len(r[i]) if isinstance(r[i], str) else 0 for r in [(h,) for h in header])) for i in range(1)]
    # fixed widths for numbers
    print("\n评估汇总 (统一含时间窗惩罚):")
    print("-" * 86)
    print(f"{header[0]:<12} | {header[1]:>12} | {header[2]:>8} | {header[3]:>12} | {header[4]:>9} | {header[5]:>9}")
    print("-" * 86)
    for name, avg, std, t, l, p in rows:
        print(f"{name:<12} | {avg:>12.2f} | {std:>8.2f} | {t:>12.3f} | {l:>9.2f} | {p:>9.2f}")
    print("-" * 86)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained VRP policy against baseline solvers.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a specific checkpoint file to evaluate. Overrides --size logic.")
    parser.add_argument("--size", type=str, default="small", choices=["small", "medium"], help="Problem size preset to use for evaluation (from config.yaml). Default: 'small'.")
    parser.add_argument("--num_instances", type=int, default=10, help="Number of random VRP instances for evaluation.")
    parser.add_argument("--decode", type=str, choices=["greedy", "sample"], default="greedy", help="Decoding strategy for policy evaluation.")
    parser.add_argument("--samples", type=int, default=1, help="Samples per instance when decode=sample (best kept).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for sampling (decode=sample).")
    parser.add_argument("--seed", type=int, default=2025, help="Base random seed for reproducibility.")
    parser.add_argument("--skip_greedy", action="store_true", help="Skip the greedy baseline.")
    parser.add_argument("--skip_ortools", action="store_true", help="Skip the OR-Tools baseline.")
    parser.add_argument("--plot_instances", type=int, default=1, help="Number of instances to plot per solver (0 to disable).")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as cfg_file:
        config = yaml.safe_load(cfg_file)

    # --- 组合最终的模型和评估配置 ---
    size_name = args.size
    if size_name not in config['problem_presets']:
        raise ValueError(f"错误: 在 config.yaml 中未找到名为 '{size_name}' 的预设规模。")

    size_config = config['problem_presets'][size_name]
    
    # 将基础参数和规模参数合并到 model_params
    model_params = {**config['model_base_params'], **size_config}
    config['model_params'] = model_params # 更新全局config，以兼容旧代码

    num_customers = model_params['num_customers']
    num_vehicles = model_params['num_vehicles']
    checkpoint_dir_name = model_params['name']

    print("\n" + "="*50)
    print(f"当前评估规模: '{size_name}'")
    print(f"  - 客户数量: {num_customers}")
    print(f"  - 车辆数量: {num_vehicles}")
    print(f"  - 解码策略: {args.decode}")
    if args.decode == 'sample':
        print(f"  - 采样次数: {args.samples}")
    print("="*50 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    trainer = Trainer(config, device)

    # 动态构建检查点路径并优先加载最佳模型
    checkpoint_dir = os.path.join("training", "checkpoints", checkpoint_dir_name)
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pth.tar")
    latest_ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pth.tar")

    load_path = args.checkpoint # 优先使用用户指定的检查点
    if load_path is None:
        if os.path.exists(best_ckpt_path):
            load_path = best_ckpt_path
            print(f"自动加载 '{size_name}' 规模的最佳模型: {load_path}")
        else:
            load_path = latest_ckpt_path
            print(f"未找到最佳模型，尝试加载 '{size_name}' 规模的最新模型: {load_path}")

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"未找到检查点文件: {load_path}")
    trainer.load_checkpoint(load_path)
    # 训练保存的 epoch 字段表示“下一轮的起始编号”，因此对应的已完成轮次为 start_epoch-1
    loaded_epoch = max(0, int(trainer.start_epoch) - 1)
    print(f"已加载检查点: {os.path.basename(load_path)} | 对应已完成的训练轮次: {loaded_epoch}")
    trainer.optimizer.zero_grad(set_to_none=True)

    print(f"\n生成 {args.num_instances} 个评估问题 (客户数: {num_customers}, 车辆数: {num_vehicles})...")
    problems = generate_problems(args.num_instances, num_customers, num_vehicles, args.seed)

    collect_details = args.plot_instances > 0

    print(f"\n评估策略网络 ({args.decode}解码)...")
    policy_result = evaluate_policy(trainer, problems, collect_details, decode=args.decode, samples=args.samples, temperature=args.temperature)

    baseline_results = evaluate_baselines(
        problems,
        use_greedy=not args.skip_greedy,
        use_or_tools=not args.skip_ortools,
        collect_details=collect_details,
    )

    if not args.skip_ortools and not _HAS_OR_TOOLS:
        print("OR-Tools 未安装，跳过该基准。")
    
    # 打印紧凑汇总表格
    _print_results_table(policy_result, baseline_results)

    if collect_details:
        plot_dir = os.path.join("evaluation_plots", checkpoint_dir_name) # 按规模保存图片
        os.makedirs(plot_dir, exist_ok=True)
        max_plots = min(args.plot_instances, len(problems))
        for idx in range(max_plots):
            problem = problems[idx]

            if "tours" in policy_result and idx < len(policy_result["tours"]):
                save_path = os.path.join(plot_dir, f"policy_instance_{idx + 1}.png")
                plot_vrp_solution(problem, policy_result["tours"][idx], f"Policy ({size_name}) - Instance {idx + 1}", save_path)

            if "greedy" in baseline_results and "tours" in baseline_results["greedy"] and idx < len(baseline_results["greedy"]["tours"]):
                save_path = os.path.join(plot_dir, f"greedy_instance_{idx + 1}.png")
                plot_vrp_solution(problem, baseline_results["greedy"]["tours"][idx], f"Greedy ({size_name}) - Instance {idx + 1}", save_path)

            if "ortools" in baseline_results and "tours" in baseline_results["ortools"] and idx < len(baseline_results["ortools"]["tours"]):
                save_path = os.path.join(plot_dir, f"ortools_instance_{idx + 1}.png")
                plot_vrp_solution(problem, baseline_results["ortools"]["tours"][idx], f"OR-Tools ({size_name}) - Instance {idx + 1}", save_path)

        print(f"\n已将解的可视化保存到 {plot_dir}/，最多 {max_plots} 个实例。")

    print("\n评估完成。")


if __name__ == "__main__":
    main()
