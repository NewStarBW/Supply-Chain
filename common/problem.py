import torch

class Problem:
    def __init__(self, num_customers, num_vehicles, customer_data, vehicle_data):
        """
        初始化一个VRP问题实例。

        参数:
            num_customers (int): 客户数量。
            num_vehicles (int): 车辆数量。
            customer_data (dict): 包含客户特征的字典。
                                  键: 'locations', 'demands', 'service_times', 'time_windows'。
            vehicle_data (dict): 包含车辆特征的字典。
                                 键: 'capacities', 'speeds'。
        """
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles

        # 客户特征
        self.locations = torch.tensor(customer_data['locations'], dtype=torch.float32)  # 包含索引为0的仓库
        self.demands = torch.tensor(customer_data['demands'], dtype=torch.float32)
        self.service_times = torch.tensor(customer_data['service_times'], dtype=torch.float32)
        self.time_windows = torch.tensor(customer_data['time_windows'], dtype=torch.float32)  # 形状: (客户数量 + 1, 2)

        # 车辆特征
        self.capacities = torch.tensor(vehicle_data['capacities'], dtype=torch.float32)
        self.speeds = torch.tensor(vehicle_data['speeds'], dtype=torch.float32)

    def get_customer_features(self):
        """
        返回一个包含所有客户组合特征的张量。
        """
        # 如果需要，可以进行特征归一化，目前仅拼接
        return torch.cat([
            self.locations,
            self.demands.unsqueeze(1),
            self.service_times.unsqueeze(1),
            self.time_windows
        ], dim=1)

    @staticmethod
    def generate_random_instance(num_customers, num_vehicles):
        """
        根据论文描述生成一个随机问题实例。
        """
        # 仓库和客户的位置在[0,1]x[0,1]范围内随机生成
        locations = torch.rand(num_customers + 1, 2)

        # 需求
        if num_vehicles == 2:
            demand_range = (1, 10)
        elif num_vehicles == 3:
            demand_range = (1, 15)
        elif num_vehicles == 4:
            demand_range = (1, 20)
        else: # 车辆数 >= 5
            demand_range = (1, 25)
        demands = torch.randint(demand_range[0], demand_range[1] + 1, (num_customers,)).float()
        demands = torch.cat([torch.tensor([0.0]), demands], dim=0) # 仓库的需求为0

        # 服务时间
        service_times = torch.rand(num_customers) * 0.1 + 0.1 # 0.1 到 0.2
        service_times = torch.cat([torch.tensor([0.0]), service_times], dim=0)

        # 时间窗
        if num_customers == 20:
            tw_start_range = (0, 4)
        elif num_customers == 50:
            tw_start_range = (0, 6)
        else: # 客户数量 == 100
            tw_start_range = (0, 8)

        tw_starts = torch.rand(num_customers) * tw_start_range[1]
        tw_lengths = torch.rand(num_customers) * 0.1 + 0.1 # 0.1 到 0.2
        tw_ends = tw_starts + tw_lengths
        time_windows = torch.stack([tw_starts, tw_ends], dim=1)
        # 仓库的时间窗很宽
        depot_tw = torch.tensor([[0.0, 1000.0]])
        time_windows = torch.cat([depot_tw, time_windows], dim=0)

        # 车辆容量
        if num_customers == 20:
            capacity = 80
        elif num_customers == 50:
            capacity = 200
        else: # 客户数量 == 100
            capacity = 400

        customer_data = {
            'locations': locations.tolist(),
            'demands': demands.tolist(),
            'service_times': service_times.tolist(),
            'time_windows': time_windows.tolist()
        }
        vehicle_data = {
            'capacities': [capacity] * num_vehicles,
            'speeds': [1.0] * num_vehicles # 为简化起见，假设速度为1
        }

        return Problem(num_customers, num_vehicles, customer_data, vehicle_data)

if __name__ == '__main__':
    # 示例用法
    problem_20_2 = Problem.generate_random_instance(20, 2)
    print("问题 (20个客户, 2辆车):")
    print("位置形状:", problem_20_2.locations.shape)
    print("需求形状:", problem_20_2.demands.shape)
    print("时间窗形状:", problem_20_2.time_windows.shape)
    print("-" * 20)

    problem_50_3 = Problem.generate_random_instance(50, 3)
    print("问题 (50个客户, 3辆车):")
    print("位置形状:", problem_50_3.locations.shape)
    print("容量:", problem_50_3.capacities)
    print("-" * 20)

    problem_100_4 = Problem.generate_random_instance(100, 4)
    print("问题 (100个客户, 4辆车):")
    print("位置形状:", problem_100_4.locations.shape)
    print("容量:", problem_100_4.capacities)
    print("-" * 20)
