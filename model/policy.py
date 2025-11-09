import torch
import torch.nn as nn
from model.encoder import Encoder
from model.recorder import RouteRecorder
from model.decoder import Decoder

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_vehicles):
        """
        主策略网络，组合了编码器、记录器和解码器。

        参数:
            input_dim (int): 每个节点的原始输入特征维度。
            embed_dim (int): 嵌入维度。
            num_heads (int): 编码器中的注意力头数。
            num_layers (int): 编码器中的注意力层数。
            num_vehicles (int): 车辆数量。
        """
        super(PolicyNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.num_vehicles = num_vehicles

        # --- 主要组件 ---
        self.encoder = Encoder(input_dim, embed_dim, num_heads, num_layers)
        self.decoder = Decoder(embed_dim, num_heads)

        # --- 路径记录器 ---
        # 每辆车的局部记录器，加入当前时间（归一化）作为额外一维
        local_recorder_input_dim = 4 # 位置(2) + 负载(1) + 当前时间(1)
        self.local_recorders = nn.ModuleList(
            [RouteRecorder(local_recorder_input_dim, embed_dim) for _ in range(num_vehicles)]
        )

        # 全局记录器
        global_recorder_input_dim = local_recorder_input_dim * num_vehicles
        self.global_recorder = RouteRecorder(global_recorder_input_dim, embed_dim)

    def forward(self, problem_features, vehicle_states, recorder_hidden_states, mask, sequential_vehicle_idx=None):
        """
        为所有车辆或单个车辆执行一个解码步骤。

        参数:
            problem_features (Tensor): 所有节点的特征。
            vehicle_states (Tensor): 所有车辆的当前状态。
            recorder_hidden_states (tuple): (local_h, global_h)。
            mask (Tensor): 车辆的掩码。
            sequential_vehicle_idx (int, optional): 如果提供，则仅为该索引的车辆解码。默认为None。

        返回:
            log_probs (Tensor): 对数概率。
            next_hidden_states (tuple): 更新后的隐藏状态。
        """
        batch_size, num_nodes, _ = problem_features.shape
        local_h, global_h = recorder_hidden_states

        # --- 编码器 ---
        node_embeddings, graph_embedding = self.encoder(problem_features)

        # --- 路径记录器更新 ---
        next_local_h_list = []
        for i in range(self.num_vehicles):
            h_prev = local_h[:, i, :]
            vehicle_state = vehicle_states[:, i, :]
            h_next = self.local_recorders[i](vehicle_state, h_prev)
            next_local_h_list.append(h_next.unsqueeze(1))
        next_local_h = torch.cat(next_local_h_list, dim=1)

        all_vehicle_states = vehicle_states.view(batch_size, -1)
        next_global_h = self.global_recorder(all_vehicle_states, global_h)

        # --- 解码器 ---
        if sequential_vehicle_idx is not None:
            # --- 顺序解码：仅为一个车辆计算 ---
            i = sequential_vehicle_idx
            observation = graph_embedding + next_local_h[:, i, :] + next_global_h
            vehicle_mask = mask[:, i, :]
            log_probs = self.decoder(observation, node_embeddings, vehicle_mask)
        else:
            # --- 并行解码：为所有车辆计算（保留用于可能的未来用途）---
            log_probs_list = []
            for i in range(self.num_vehicles):
                observation = graph_embedding + next_local_h[:, i, :] + next_global_h
                vehicle_mask = mask[:, i, :]
                lp = self.decoder(observation, node_embeddings, vehicle_mask)
                log_probs_list.append(lp.unsqueeze(1))
            log_probs = torch.cat(log_probs_list, dim=1)

        return log_probs, (next_local_h, next_global_h)

if __name__ == '__main__':
    # 超参数
    INPUT_DIM = 6 # 位置(2)+需求(1)+服务时间(1)+时间窗(2)
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 3
    NUM_VEHICLES = 3
    BATCH_SIZE = 4
    NUM_NODES = 21

    policy_net = PolicyNetwork(INPUT_DIM, EMBED_DIM, NUM_HEADS, NUM_LAYERS, NUM_VEHICLES)

    # 虚拟输入
    dummy_features = torch.rand(BATCH_SIZE, NUM_NODES, INPUT_DIM)
    dummy_vehicle_states = torch.rand(BATCH_SIZE, NUM_VEHICLES, 3)
    dummy_local_h = torch.zeros(BATCH_SIZE, NUM_VEHICLES, EMBED_DIM)
    dummy_global_h = torch.zeros(BATCH_SIZE, EMBED_DIM)
    dummy_mask = torch.zeros(BATCH_SIZE, NUM_VEHICLES, NUM_NODES, dtype=torch.bool)

    log_probs, (next_local_h, next_global_h) = policy_net(
        dummy_features,
        dummy_vehicle_states,
        (dummy_local_h, dummy_global_h),
        dummy_mask
    )

    print("策略网络测试:")
    print("输入特征形状:", dummy_features.shape)
    print("输入车辆状态形状:", dummy_vehicle_states.shape)
    print("输出对数概率形状:", log_probs.shape)
    print("下一个局部隐藏状态形状:", next_local_h.shape)
    print("下一个全局隐藏状态形状:", next_global_h.shape)

    assert log_probs.shape == (BATCH_SIZE, NUM_VEHICLES, NUM_NODES)
    assert next_local_h.shape == (BATCH_SIZE, NUM_VEHICLES, EMBED_DIM)
    assert next_global_h.shape == (BATCH_SIZE, EMBED_DIM)
    print("测试通过!")
