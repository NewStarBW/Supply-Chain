
import time
import torch
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_or_tools(problem):
    """
    Solves the VRP using Google's OR-Tools.

    Args:
        problem (VRPProblem): The VRP problem instance from our project.

    Returns:
        A tuple containing:
        - total_cost (float): The total distance of all routes.
        - tours (list of lists): The list of tours for each vehicle.
        - duration (float): The time taken to solve the problem.
    """
    start_time = time.time()

    # --- 1. Data Preparation ---
    # Extract data from the PyTorch-based problem instance
    locations = problem.locations.cpu().numpy()
    demands = problem.demands.cpu().numpy()
    num_vehicles = problem.num_vehicles
    depot_idx = 0
    
    # OR-Tools works best with integer costs. We scale float distances by a large factor.
    scaling_factor = 10000
    dist_matrix = torch.cdist(problem.locations, problem.locations, p=2).cpu().numpy()
    dist_matrix_int = (dist_matrix * scaling_factor).astype(int)

    # --- 2. Create OR-Tools Data Model ---
    data = {}
    data['distance_matrix'] = dist_matrix_int
    data['demands'] = demands.astype(int)
    data['vehicle_capacities'] = problem.capacities.cpu().numpy().astype(int)
    data['num_vehicles'] = num_vehicles
    data['depot'] = depot_idx

    # --- 3. Setup Routing Model ---
    # Create the routing index manager and the routing model.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # --- 4. Register Callbacks and Constraints ---
    # Create and register a transit callback (distance callback).
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint.
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # --- 5. Set Search Parameters and Solve ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(5) # Set a time limit for the search

    solution = routing.SolveWithParameters(search_parameters)

    # --- 6. Parse and Return Solution ---
    if solution:
        total_cost = solution.ObjectiveValue() / scaling_factor # Rescale cost
        tours = []
        for vehicle_id in range(data['num_vehicles']):
            tour = []
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != data['depot']:
                    tour.append(node_index)
                index = solution.Value(routing.NextVar(index))
            tours.append(tour)
        
        duration = time.time() - start_time
        return total_cost, tours, duration
    else:
        # No solution found
        return float('inf'), [], time.time() - start_time
