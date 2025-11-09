import torch
from common.problem import Problem

class Env:
    def __init__(self, problem):
        """
        初始化VRP环境。

        参数:
            problem (Problem): VRP问题实例。
        """
        self.problem = problem
        self.num_nodes = problem.num_customers + 1
        self.num_vehicles = problem.num_vehicles

        # 动态状态
        self.reset()

    def reset(self):
        """
        将环境重置到初始状态。
        """
        # 每辆车的路径，从仓库（节点0）开始
        self.routes = [[0] for _ in range(self.num_vehicles)]

        # 每辆车的当前位置
        self.current_locations = torch.zeros(self.num_vehicles, dtype=torch.long)

        # 每辆车的剩余容量
        self.remaining_capacities = self.problem.capacities.clone()

        # 每辆车的当前时间
        self.current_times = torch.zeros(self.num_vehicles, dtype=torch.float32)

        # 已访问客户的掩码（1表示已访问，0表示未访问）
        self.visited_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.visited_mask[0] = True # 仓库在某种意义上初始即为“已访问”

        # 每辆车路径上每个节点的到达时间
        self.arrival_times = [[] for _ in range(self.num_vehicles)]

        # 跟踪哪些车辆已完成其路径
        self.finished_vehicles = torch.zeros(self.num_vehicles, dtype=torch.bool)

    def step(self, vehicle_idx, next_node):
        """
        在车辆移动到下一个节点后更新环境状态。

        参数:
            vehicle_idx (int): 正在移动的车辆索引。
            next_node (int): 要访问的下一个节点的索引。
        """
        if self.finished_vehicles[vehicle_idx]:
            return # 该车辆已经返回仓库

        prev_node = self.current_locations[vehicle_idx].item()

        # 更新时间
        travel_time = self._get_travel_time(prev_node, next_node, vehicle_idx)

        # 车辆在上一个节点变为空闲的时间点
        free_time_at_prev_node = self.current_times[vehicle_idx] + self.problem.service_times[prev_node]

        arrival_time = free_time_at_prev_node + travel_time

        # 存储精确的到达时间以用于成本计算
        self.arrival_times[vehicle_idx].append(arrival_time.item())

        # 更新车辆的当前时间。这是车辆在新节点上的到达时间。
        # 论文指出“车辆不允许在客户节点等待”，所以车辆的新的当前时间就是它在新节点的到达时间。
        self.current_times[vehicle_idx] = arrival_time

        # 更新位置和路径
        self.current_locations[vehicle_idx] = next_node
        self.routes[vehicle_idx].append(next_node)

        if next_node != 0: # 如果不是返回仓库
            # 更新容量
            self.remaining_capacities[vehicle_idx] -= self.problem.demands[next_node]
            # 标记客户为已访问
            self.visited_mask[next_node] = True
        else: # 如果返回仓库
            self.finished_vehicles[vehicle_idx] = True

    def get_mask(self, vehicle_idx):
        """
        为给定车辆获取一个有效下一节点的掩码。

        参数:
            vehicle_idx (int): 车辆的索引。

        返回:
            一个布尔张量，其中True表示一个无效（被遮蔽）的节点。
        """
        mask = self.visited_mask.clone()

        # 遮蔽超过车辆容量的节点
        for i in range(1, self.num_nodes): # 排除仓库
            if self.problem.demands[i] > self.remaining_capacities[vehicle_idx]:
                mask[i] = True

        # 仓库（节点0）的遮蔽逻辑，来自论文（公式4b）
        # 如果仍有可访问的客户，则遮蔽仓库
        can_visit_more = False
        for i in range(1, self.num_nodes):
            if not mask[i]: # 如果至少有一个未访问且容量足够的客户
                can_visit_more = True
                break

        if can_visit_more:
            mask[0] = True # 在返回仓库前必须访问客户
        else:
            mask[0] = False # 可以返回仓库

        return mask

    def all_finished(self):
        """
        检查是否所有客户都已被访问，或所有车辆都已返回仓库。
        """
        return self.visited_mask[1:].all() or self.finished_vehicles.all()

    def _get_travel_time(self, node1, node2, vehicle_idx):
        """计算特定车辆在两个节点之间的行驶时间。"""
        dist = torch.norm(self.problem.locations[node1] - self.problem.locations[node2], p=2)
        return dist / self.problem.speeds[vehicle_idx]

    def calculate_costs(self):
        """计算当前解（路径）的总成本及其构成。

        返回:
            total_cost (float): 路径总长度 + 时间窗惩罚。
            total_length (float): 路径总长度。
            total_penalty (float): 时间窗惩罚（早到 + 晚到）。
            total_early (float): 早到惩罚总和。
            total_late (float): 晚到惩罚总和。
        """
        total_length = 0.0
        total_penalty = 0.0
        total_early = 0.0
        total_late = 0.0

        alpha = 0.5  # 早到惩罚系数
        beta = 2.0   # 晚到惩罚系数

        for i in range(self.num_vehicles):
            route = self.routes[i]
            if len(route) <= 1:  # 空路径或只有仓库
                continue

            # --- 计算路径长度 ---
            if len(route) > 1:
                route_locations = self.problem.locations[route]
                total_length += torch.sum(
                    torch.norm(route_locations[1:] - route_locations[:-1], p=2, dim=1)
                )

            # --- 使用预先记录的到达时间计算时间惩罚 ---
            customer_route_part = route[1:]  # 去掉起始仓库
            for j, node_idx in enumerate(customer_route_part):
                if node_idx == 0:  # 返回仓库不计算惩罚
                    continue
                arrival_time = self.arrival_times[i][j]
                e_j, l_j = self.problem.time_windows[node_idx]
                early_penalty = torch.max(torch.tensor(0.0), e_j - arrival_time) * alpha
                late_penalty = torch.max(torch.tensor(0.0), arrival_time - l_j) * beta
                total_penalty += early_penalty + late_penalty
                total_early += early_penalty
                total_late += late_penalty

        total_cost = total_length + total_penalty
        return (
            float(total_cost),
            float(total_length),
            float(total_penalty),
            float(total_early),
            float(total_late),
        )

if __name__ == '__main__':
    # 示例用法
    problem = Problem.generate_random_instance(20, 2)
    env = Env(problem)

    print("初始状态:")
    print("路径:", env.routes)
    print("已访问掩码:", env.visited_mask)
    print("车辆0的初始掩码:", env.get_mask(0))

    # 模拟车辆0的几步
    env.step(0, 5)
    env.step(0, 12)

    # 模拟车辆1的几步
    env.step(1, 3)

    print("\n几步之后的状态:")
    print("路径:", env.routes)
    print("已访问掩码:", env.visited_mask)
    print("车辆0的掩码:", env.get_mask(0))
    print("剩余容量:", env.remaining_capacities)
    print("当前时间:", env.current_times)

    # 模拟返回仓库
    env.step(0, 0)
    print("\n车辆0返回仓库后的状态:")
    print("已完成车辆:", env.finished_vehicles)
    print("车辆0的掩码:", env.get_mask(0))

    cost, length, penalty, early_p, late_p = env.calculate_costs()
    print(
        f"\n最终计算成本: {cost:.2f} (长度: {length:.2f}, 总惩罚: {penalty:.2f}, 早到: {early_p:.2f}, 晚到: {late_p:.2f})"
    )
