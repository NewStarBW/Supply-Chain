import torch
import torch.nn as nn

class RouteRecorder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        """
        使用GRU单元的路径记录器模块。
        既可以用作局部记录器，也可以用作全局记录器。

        参数:
            input_dim (int): 每一步输入的维度。
                             对于局部记录器: 车辆状态 (例如, 位置 + 负载)。
                             对于全局记录器: 所有车辆状态的组合。
            embed_dim (int): 嵌入维度 (隐藏状态大小)。
        """
        super(RouteRecorder, self).__init__()
        self.gru_cell = nn.GRUCell(input_dim, embed_dim)

    def forward(self, x, h_prev):
        """
        单步前向传播。

        参数:
            x (Tensor): 当前步骤的输入张量，形状为 (批次大小, 输入维度)。
            h_prev (Tensor): 上一个隐藏状态，形状为 (批次大小, 嵌入维度)。

        返回:
            h_next (Tensor): 下一个隐藏状态，形状为 (批次大小, 嵌入维度)。
        """
        h_next = self.gru_cell(x, h_prev)
        return h_next

if __name__ == '__main__':
    # 超参数
    EMBED_DIM = 128
    BATCH_SIZE = 4
    NUM_VEHICLES = 3

    # --- 局部路径记录器测试 ---
    # 输入: 当前位置 (2D) + 剩余负载 (1D) = 3
    local_input_dim = 3
    local_recorder = RouteRecorder(local_input_dim, EMBED_DIM)

    # 单个车辆的虚拟输入
    dummy_vehicle_state = torch.rand(BATCH_SIZE, local_input_dim)
    # 初始隐藏状态 (通常为零)
    h_prev_local = torch.zeros(BATCH_SIZE, EMBED_DIM)

    h_next_local = local_recorder(dummy_vehicle_state, h_prev_local)

    print("局部路径记录器测试:")
    print("输入状态形状:", dummy_vehicle_state.shape)
    print("上一个隐藏状态形状:", h_prev_local.shape)
    print("下一个隐藏状态形状:", h_next_local.shape)
    assert h_next_local.shape == (BATCH_SIZE, EMBED_DIM)
    print("测试通过!")
    print("-" * 20)

    # --- 全局路径记录器测试 ---
    # 输入: 所有车辆状态的拼接
    global_input_dim = local_input_dim * NUM_VEHICLES
    global_recorder = RouteRecorder(global_input_dim, EMBED_DIM)

    # 所有车辆组合状态的虚拟输入
    dummy_all_vehicles_state = torch.rand(BATCH_SIZE, global_input_dim)
    # 初始隐藏状态
    h_prev_global = torch.zeros(BATCH_SIZE, EMBED_DIM)

    h_next_global = global_recorder(dummy_all_vehicles_state, h_prev_global)

    print("全局路径记录器测试:")
    print("输入状态形状:", dummy_all_vehicles_state.shape)
    print("上一个隐藏状态形状:", h_prev_global.shape)
    print("下一个隐藏状态形状:", h_next_global.shape)
    assert h_next_global.shape == (BATCH_SIZE, EMBED_DIM)
    print("测试通过!")
