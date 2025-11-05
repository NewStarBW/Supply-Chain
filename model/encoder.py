import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        """
        Encoder module based on Multi-Head Attention.

        Args:
            input_dim (int): Dimension of the raw input features for each node.
            embed_dim (int): The embedding dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of attention layers.
        """
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.initial_projection = nn.Linear(input_dim, embed_dim)

        self.attention_layers = nn.ModuleList([
            MultiHeadAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_nodes, input_dim).

        Returns:
            node_embeddings (Tensor): Embeddings for each node, shape (batch_size, num_nodes, embed_dim).
            graph_embedding (Tensor): Embedding for the entire graph, shape (batch_size, embed_dim).
        """
        # Initial projection
        h = self.initial_projection(x) # (batch_size, num_nodes, embed_dim)

        # Pass through attention layers
        for layer in self.attention_layers:
            h = layer(h)

        # Final node embeddings
        node_embeddings = h

        # Graph embedding (context vector)
        graph_embedding = h.mean(dim=1)

        return node_embeddings, graph_embedding


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        A single layer of Multi-Head Attention followed by a Feed-Forward network.
        Corresponds to equations (6), (7), (8) in the paper.
        """
        super(MultiHeadAttentionLayer, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim)

        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

    def forward(self, h):
        """
        Args:
            h (Tensor): Input tensor of shape (batch_size, num_nodes, embed_dim).

        Returns:
            Tensor of shape (batch_size, num_nodes, embed_dim).
        """
        batch_size, num_nodes, embed_dim = h.shape

        # Multi-Head Attention with skip connection and batch norm
        h_res = h
        h_mha = self.mha(h)

        # Reshape for BatchNorm1d: (batch_size * num_nodes, embed_dim)
        h = h_res.view(-1, embed_dim) + h_mha.view(-1, embed_dim)
        h = self.bn1(h).view(batch_size, num_nodes, embed_dim)

        # Feed-Forward with skip connection and batch norm
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

        # Reshape and transpose for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)

        # Concatenate heads and apply final linear layer
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
    # Hyperparameters from the paper
    INPUT_DIM = 7 # Location(2) + Demand(1) + ServiceTime(1) + TimeWindow(2) + is_depot(1)
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 3

    # Example usage
    encoder = Encoder(input_dim=INPUT_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)

    # Dummy input: batch_size=4, num_nodes=21 (1 depot + 20 customers)
    dummy_input = torch.rand(4, 21, INPUT_DIM)

    node_embeds, graph_embed = encoder(dummy_input)

    print("Encoder test:")
    print("Input shape:", dummy_input.shape)
    print("Node embeddings shape:", node_embeds.shape)
    print("Graph embedding shape:", graph_embed.shape)
    assert node_embeds.shape == (4, 21, EMBED_DIM)
    assert graph_embed.shape == (4, EMBED_DIM)
    print("Test passed!")

