import torch
import torch.nn as nn
import math

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        """
        解码器模块，用于选择下一个节点。

        参数:
            embed_dim (int): 嵌入维度。
            num_heads (int): 注意力机制的头数。
        """
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim

        # 论文中简化的注意力机制 (公式 13)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)

        # 用于多头注意力的兼容性 (可选，但是个好习惯)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, observation, node_embeddings, mask):
        """
        计算接下来访问每个节点的概率。

        参数:
            observation (Tensor): 当前车辆的观测向量，
                                  形状为 (批次大小, 嵌入维度)。
            node_embeddings (Tensor): 编码器输出的所有节点的嵌入，
                                      形状为 (批次大小, 节点数量, 嵌入维度)。
            mask (Tensor): 一个布尔掩码，其中True表示无效节点，
                           形状为 (批次大小, 节点数量)。

        返回:
            log_probs (Tensor): 每个节点的对数概率，形状为 (批次大小, 节点数量)。
        """
        batch_size, num_nodes, _ = node_embeddings.shape

        # 公式 13a: 将观测向量投影为查询(query)
        query = self.W_q(observation) # (批次大小, 嵌入维度)

        # 公式 13b: 将节点嵌入投影为键(keys)
        keys = self.W_k(node_embeddings) # (批次大小, 节点数量, 嵌入维度)

        # 公式 13c: 计算兼容性得分 (u_i,j,t)
        # 按照论文文本描述使用单头注意力
        query = query.unsqueeze(1) # (批次大小, 1, 嵌入维度)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        scores = scores.squeeze(1) # (批次大小, 节点数量)

        # 在softmax之前应用掩码
        scores[mask] = -float('inf')

        # 公式 14: 计算概率 (softmax) 并返回对数概率
        log_probs = torch.log_softmax(scores, dim=-1)

        return log_probs

if __name__ == '__main__':
    # 超参数
    EMBED_DIM = 128
    BATCH_SIZE = 4
    NUM_NODES = 21 # 1个仓库 + 20个客户

    decoder = Decoder(EMBED_DIM)

    # 虚拟输入
    # 当前车辆的观测向量
    dummy_observation = torch.rand(BATCH_SIZE, EMBED_DIM)
    # 编码器输出的节点嵌入
    dummy_node_embeddings = torch.rand(BATCH_SIZE, NUM_NODES, EMBED_DIM)
    # 掩码: 假设第一个批次项的节点3和5是无效的
    dummy_mask = torch.zeros(BATCH_SIZE, NUM_NODES, dtype=torch.bool)
    dummy_mask[0, 3] = True
    dummy_mask[0, 5] = True

    log_probs = decoder(dummy_observation, dummy_node_embeddings, dummy_mask)

    print("解码器测试:")
    print("观测向量形状:", dummy_observation.shape)
    print("节点嵌入形状:", dummy_node_embeddings.shape)
    print("掩码形状:", dummy_mask.shape)
    print("输出对数概率形状:", log_probs.shape)
    assert log_probs.shape == (BATCH_SIZE, NUM_NODES)

    # 检查被遮蔽的概率是否为-inf
    print("被遮蔽节点[0, 3]的对数概率:", log_probs[0, 3].item())
    assert log_probs[0, 3].item() == -float('inf')

    # 检查概率总和是否为1
    probs = torch.exp(log_probs)
    print("第一个批次项的概率总和:", probs[0].sum().item())
    assert math.isclose(probs[0].sum().item(), 1.0, rel_tol=1e-6)

    print("测试通过!")
