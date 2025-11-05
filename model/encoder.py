import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        """
        基于多头注意力的编码器模块。

        参数:
            input_dim (int): 每个节点的原始输入特征维度。
            embed_dim (int): 嵌入维度。
            num_heads (int): 注意力头的数量。
            num_layers (int): 注意力层的数量。
        """
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.initial_projection = nn.Linear(input_dim, embed_dim)

        self.attention_layers = nn.ModuleList([
            MultiHeadAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        编码器的前向传播。

        参数:
            x (Tensor): 输入张量，形状为 (批次大小, 节点数量, 输入维度)。

        返回:
            node_embeddings (Tensor): 每个节点的嵌入，形状为 (批次大小, 节点数量, 嵌入维度)。
            graph_embedding (Tensor): 整个图的嵌入，形状为 (批次大小, 嵌入维度)。
        """
        # 初始投影
        h = self.initial_projection(x) # (批次大小, 节点数量, 嵌入维度)

        # 通过注意力层
        for layer in self.attention_layers:
            h = layer(h)

        # 最终的节点嵌入
        node_embeddings = h

        # 图嵌入 (上下文向量)
        graph_embedding = h.mean(dim=1)

        return node_embeddings, graph_embedding


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        一个多头注意力层，后跟一个前馈网络。
        对应论文中的公式 (6), (7), (8)。
        """
        super(MultiHeadAttentionLayer, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim)

        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

    def forward(self, h):
        """
        参数:
            h (Tensor): 输入张量，形状为 (批次大小, 节点数量, 嵌入维度)。

        返回:
            形状为 (批次大小, 节点数量, 嵌入维度) 的张量。
        """
        batch_size, num_nodes, embed_dim = h.shape

        # 带跳跃连接和批归一化的多头注意力
        h_res = h
        h_mha = self.mha(h)

        # 为BatchNorm1d重塑形状: (批次大小 * 节点数量, 嵌入维度)
        h = h_res.view(-1, embed_dim) + h_mha.view(-1, embed_dim)
        h = self.bn1(h).view(batch_size, num_nodes, embed_dim)

        # 带跳跃连接和批归一化的前馈网络
        h_res = h
        h_ff = self.ff(h)

        h = h_res.view(-1, embed_dim) + h_ff.view(-1, embed_dim)
        h = self.bn2(h).view(batch_size, num_nodes, embed_dim)

        return h


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, h):
        batch_size, num_nodes, embed_dim = h.shape

        Q = self.W_q(h)
        K = self.W_k(h)
        V = self.W_v(h)

        # 为多头注意力重塑和转置
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)

        # 拼接多头并应用最终的线性层
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, embed_dim)

        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim=512):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    # 论文中的超参数
    INPUT_DIM = 7 # 位置(2) + 需求(1) + 服务时间(1) + 时间窗(2) + 是否是仓库(1)
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 3

    # 示例用法
    encoder = Encoder(input_dim=INPUT_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)

    # 虚拟输入: 批次大小=4, 节点数量=21 (1个仓库 + 20个客户)
    dummy_input = torch.rand(4, 21, INPUT_DIM)

    node_embeds, graph_embed = encoder(dummy_input)

    print("编码器测试:")
    print("输入形状:", dummy_input.shape)
    print("节点嵌入形状:", node_embeds.shape)
    print("图嵌入形状:", graph_embed.shape)
    assert node_embeds.shape == (4, 21, EMBED_DIM)
    assert graph_embed.shape == (4, EMBED_DIM)
    print("测试通过!")
