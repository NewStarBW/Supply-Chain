import torch
from common.problem import Problem

class Env:
    def __init__(self, problem):
        """
        Initializes the VRP environment.

        Args:
            problem (Problem): The VRP problem instance.
        """
        self.problem = problem
        self.num_nodes = problem.num_customers + 1
        self.num_vehicles = problem.num_vehicles

        # Dynamic state
        self.reset()

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        # Routes for each vehicle, starting at the depot (node 0)
        self.routes = [[0] for _ in range(self.num_vehicles)]

        # Current location of each vehicle
        self.current_locations = torch.zeros(self.num_vehicles, dtype=torch.long)

        # Remaining capacity of each vehicle
        self.remaining_capacities = self.problem.capacities.clone()

        # Current time of each vehicle
        self.current_times = torch.zeros(self.num_vehicles, dtype=torch.float32)

        # Mask for visited customers (1 for visited, 0 for not visited)
        self.visited_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.visited_mask[0] = True # Depot is initially "visited" in a sense

        # Arrival time at each node for each vehicle's route
        self.arrival_times = [[] for _ in range(self.num_vehicles)]

        # Keep track of which vehicles have finished their routes
        self.finished_vehicles = torch.zeros(self.num_vehicles, dtype=torch.bool)

    def step(self, vehicle_idx, next_node):
        """
        Updates the environment state after a vehicle moves to the next node.

        Args:
            vehicle_idx (int): The index of the vehicle that is moving.
            next_node (int): The index of the next node to visit.
        """
        if self.finished_vehicles[vehicle_idx]:
            return # This vehicle has already returned to the depot

        prev_node = self.current_locations[vehicle_idx].item()

        # Update time
        travel_time = self._get_travel_time(prev_node, next_node, vehicle_idx)

        # The time at which the vehicle becomes free at the previous node
        free_time_at_prev_node = self.current_times[vehicle_idx] + self.problem.service_times[prev_node]

        arrival_time = free_time_at_prev_node + travel_time

        # Store the precise arrival time for cost calculation
        self.arrival_times[vehicle_idx].append(arrival_time.item())

        # Update vehicle's current time. This is the time the vehicle is free to leave the *previous* node.
        # The paper states "vehicles are not allowed to wait", so the new current_time for the vehicle
        # is the arrival time at the new node.
        self.current_times[vehicle_idx] = arrival_time

        # Update location and route
        self.current_locations[vehicle_idx] = next_node
        self.routes[vehicle_idx].append(next_node)

        if next_node != 0: # If not returning to depot
            # Update capacity
            self.remaining_capacities[vehicle_idx] -= self.problem.demands[next_node]
            # Mark customer as visited
            self.visited_mask[next_node] = True
        else: # If returning to depot
            self.finished_vehicles[vehicle_idx] = True

    def get_mask(self, vehicle_idx):
        """
        Gets a mask of valid next nodes for a given vehicle.

        Args:
            vehicle_idx (int): The index of the vehicle.

        Returns:
            A boolean tensor where True indicates an invalid (masked) node.
        """
        mask = self.visited_mask.clone()

        # Mask nodes that exceed vehicle capacity
        for i in range(1, self.num_nodes): # Exclude depot
            if self.problem.demands[i] > self.remaining_capacities[vehicle_idx]:
                mask[i] = True

        # Depot (node 0) masking logic from paper (Eq. 4b)
        # Mask depot if there are still visitable customers
        can_visit_more = False
        for i in range(1, self.num_nodes):
            if not mask[i]: # If there is at least one unvisited, capacitated customer
                can_visit_more = True
                break

        if can_visit_more:
            mask[0] = True # Must visit customers before returning to depot
        else:
            mask[0] = False # Can return to depot

        return mask

    def all_finished(self):
        """
        Checks if all customers have been visited or all vehicles have returned to the depot.
        """
        return self.visited_mask[1:].all() or self.finished_vehicles.all()

    def _get_travel_time(self, node1, node2, vehicle_idx):
        """Calculates travel time between two nodes for a specific vehicle."""
        dist = torch.norm(self.problem.locations[node1] - self.problem.locations[node2], p=2)
        return dist / self.problem.speeds[vehicle_idx]

    def calculate_costs(self):
        """
        Calculates the total cost of the current solution (routes).
        Follows equations (1), (2), and (3) from the paper, using precise timings.
        """
        total_length = 0.0
        total_penalty = 0.0

        alpha = 0.5 # Early arrival penalty coefficient
        beta = 2.0  # Late arrival penalty coefficient

        for i in range(self.num_vehicles):
            route = self.routes[i]
            if len(route) <= 1: # Empty or just depot
                continue

            # --- Calculate path length ---
            # Ensure route has more than one node to calculate length
            if len(route) > 1:
                route_locations = self.problem.locations[route]
                total_length += torch.sum(torch.norm(route_locations[1:] - route_locations[:-1], p=2, dim=1))

            # --- Calculate time penalty using pre-recorded arrival times ---
            # The route list includes the start depot, but arrival_times list does not,
            # as we only record arrivals at the *next* node.
            # So, arrival_times[i][j] corresponds to the arrival at route[j+1].
            customer_route_part = route[1:] # Nodes visited after depot

            for j, node_idx in enumerate(customer_route_part):
                if node_idx == 0: # Skip penalty for returning to depot
                    continue

                arrival_time = self.arrival_times[i][j]

                e_j, l_j = self.problem.time_windows[node_idx]

                early_penalty = torch.max(torch.tensor(0.0), e_j - arrival_time) * alpha
                late_penalty = torch.max(torch.tensor(0.0), arrival_time - l_j) * beta
                total_penalty += early_penalty + late_penalty

        total_cost = total_length + total_penalty
        return total_cost, total_length, total_penalty

if __name__ == '__main__':
    # Example usage
    problem = Problem.generate_random_instance(20, 2)
    env = Env(problem)

    print("Initial state:")
    print("Routes:", env.routes)
    print("Visited mask:", env.visited_mask)
    print("Initial mask for vehicle 0:", env.get_mask(0))

    # Simulate a few steps for vehicle 0
    env.step(0, 5)
    env.step(0, 12)

    # Simulate a few steps for vehicle 1
    env.step(1, 3)

    print("\nState after a few steps:")
    print("Routes:", env.routes)
    print("Visited mask:", env.visited_mask)
    print("Mask for vehicle 0:", env.get_mask(0))
    print("Remaining capacities:", env.remaining_capacities)
    print("Current times:", env.current_times)

    # Simulate returning to depot
    env.step(0, 0)
    print("\nState after vehicle 0 returns to depot:")
    print("Finished vehicles:", env.finished_vehicles)
    print("Mask for vehicle 0:", env.get_mask(0))

    cost, length, penalty = env.calculate_costs()
    print(f"\nFinal calculated cost: {cost:.2f} (Length: {length:.2f}, Penalty: {penalty:.2f})")
