import torch

class Problem:
    def __init__(self, num_customers, num_vehicles, customer_data, vehicle_data):
        """
        Initializes a VRP problem instance.

        Args:
            num_customers (int): Number of customers.
            num_vehicles (int): Number of vehicles.
            customer_data (dict): A dictionary containing customer features.
                                  Keys: 'locations', 'demands', 'service_times', 'time_windows'.
            vehicle_data (dict): A dictionary containing vehicle features.
                                 Keys: 'capacities', 'speeds'.
        """
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles

        # Customer features
        self.locations = torch.tensor(customer_data['locations'], dtype=torch.float32)  # Includes depot at index 0
        self.demands = torch.tensor(customer_data['demands'], dtype=torch.float32)
        self.service_times = torch.tensor(customer_data['service_times'], dtype=torch.float32)
        self.time_windows = torch.tensor(customer_data['time_windows'], dtype=torch.float32)  # Shape: (num_customers + 1, 2)

        # Vehicle features
        self.capacities = torch.tensor(vehicle_data['capacities'], dtype=torch.float32)
        self.speeds = torch.tensor(vehicle_data['speeds'], dtype=torch.float32)

    def get_customer_features(self):
        """
        Returns a tensor of all customer features combined.
        """
        # Normalize features if necessary, for now, just concatenate
        return torch.cat([
            self.locations,
            self.demands.unsqueeze(1),
            self.service_times.unsqueeze(1),
            self.time_windows
        ], dim=1)

    @staticmethod
    def generate_random_instance(num_customers, num_vehicles):
        """
        Generates a random problem instance based on the paper's description.
        """
        # Depot and customer locations are randomly generated in [0,1]x[0,1]
        locations = torch.rand(num_customers + 1, 2)

        # Demands
        if num_vehicles == 2:
            demand_range = (1, 10)
        elif num_vehicles == 3:
            demand_range = (1, 15)
        elif num_vehicles == 4:
            demand_range = (1, 20)
        else: # num_vehicles >= 5
            demand_range = (1, 25)
        demands = torch.randint(demand_range[0], demand_range[1] + 1, (num_customers,)).float()
        demands = torch.cat([torch.tensor([0.0]), demands], dim=0) # Depot demand is 0

        # Service times
        service_times = torch.rand(num_customers) * 0.1 + 0.1 # 0.1 to 0.2
        service_times = torch.cat([torch.tensor([0.0]), service_times], dim=0)

        # Time windows
        if num_customers == 20:
            tw_start_range = (0, 4)
        elif num_customers == 50:
            tw_start_range = (0, 6)
        else: # num_customers == 100
            tw_start_range = (0, 8)

        tw_starts = torch.rand(num_customers) * tw_start_range[1]
        tw_lengths = torch.rand(num_customers) * 0.1 + 0.1 # 0.1 to 0.2
        tw_ends = tw_starts + tw_lengths
        time_windows = torch.stack([tw_starts, tw_ends], dim=1)
        # Depot has a wide time window
        depot_tw = torch.tensor([[0.0, 1000.0]])
        time_windows = torch.cat([depot_tw, time_windows], dim=0)

        # Vehicle capacity
        if num_customers == 20:
            capacity = 80
        elif num_customers == 50:
            capacity = 200
        else: # num_customers == 100
            capacity = 400

        customer_data = {
            'locations': locations.tolist(),
            'demands': demands.tolist(),
            'service_times': service_times.tolist(),
            'time_windows': time_windows.tolist()
        }
        vehicle_data = {
            'capacities': [capacity] * num_vehicles,
            'speeds': [1.0] * num_vehicles # Assume speed is 1 for simplicity
        }

        return Problem(num_customers, num_vehicles, customer_data, vehicle_data)

if __name__ == '__main__':
    # Example usage
    problem_20_2 = Problem.generate_random_instance(20, 2)
    print("Problem (20 customers, 2 vehicles):")
    print("Locations shape:", problem_20_2.locations.shape)
    print("Demands shape:", problem_20_2.demands.shape)
    print("Time windows shape:", problem_20_2.time_windows.shape)
    print("-" * 20)

    problem_50_3 = Problem.generate_random_instance(50, 3)
    print("Problem (50 customers, 3 vehicles):")
    print("Locations shape:", problem_50_3.locations.shape)
    print("Capacities:", problem_50_3.capacities)
    print("-" * 20)

    problem_100_4 = Problem.generate_random_instance(100, 4)
    print("Problem (100 customers, 4 vehicles):")
    print("Locations shape:", problem_100_4.locations.shape)
    print("Capacities:", problem_100_4.capacities)
    print("-" * 20)

