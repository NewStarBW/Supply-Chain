
import torch
import time

def solve_greedy(problem):
    """
    Solves the VRP using a simple nearest-neighbor greedy heuristic.

    Args:
        problem (VRPProblem): The VRP problem instance.

    Returns:
        A tuple containing:
        - total_cost (float): The total distance of all routes.
        - tours (list of lists): The list of tours for each vehicle.
        - duration (float): The time taken to solve the problem.
    """
    start_time = time.time()
    
    device = problem.device
    num_nodes = problem.num_nodes + 1
    num_vehicles = problem.num_vehicles

    # --- State Tracking ---
    tours = [[] for _ in range(num_vehicles)]
    visited_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    visited_mask[0] = True # Depot is always 'visited'

    current_nodes = torch.zeros(num_vehicles, dtype=torch.long, device=device)
    remaining_capacities = problem.capacities.clone()

    active_vehicles = torch.ones(num_vehicles, dtype=torch.bool, device=device)

    # --- Main Loop ---
    # Continue as long as there are unvisited customers
    while not visited_mask[1:].all():
        
        # Check if any vehicle can continue
        if not active_vehicles.any():
            break # All vehicles have returned to depot

        for i in range(num_vehicles):
            if not active_vehicles[i]:
                continue

            current_node = current_nodes[i]
            
            # --- Find the best next node for the current vehicle ---
            best_next_node = -1
            min_dist = float('inf')

            # Create a distance matrix from the current node to all other nodes
            dists = torch.norm(problem.locations[current_node] - problem.locations, p=2, dim=1)
            
            # Iterate through all potential next nodes
            for next_node_idx in range(1, num_nodes): # Only consider customers
                # Check validity:
                # 1. Not already visited
                # 2. Demand does not exceed remaining capacity
                if not visited_mask[next_node_idx] and problem.demands[next_node_idx] <= remaining_capacities[i]:
                    if dists[next_node_idx] < min_dist:
                        min_dist = dists[next_node_idx]
                        best_next_node = next_node_idx
            
            # --- Update State ---
            if best_next_node != -1:
                # Move to the best found node
                tours[i].append(best_next_node)
                visited_mask[best_next_node] = True
                current_nodes[i] = best_next_node
                remaining_capacities[i] -= problem.demands[best_next_node]
            else:
                # No valid customer node found, vehicle must return to depot
                active_vehicles[i] = False

    # --- Calculate Final Cost ---
    total_cost = 0.0
    for i, tour in enumerate(tours):
        if not tour:
            continue
        
        # Add depot at start and end for cost calculation
        full_route = [0] + tour + [0]
        route_locs = problem.locations[full_route]
        total_cost += torch.sum(torch.norm(route_locs[1:] - route_locs[:-1], p=2, dim=1)).item()

    duration = time.time() - start_time
    
    return total_cost, tours, duration
