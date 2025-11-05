import torch
import torch.nn as nn
import math

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        """
        Decoder module to select the next node.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): Number of heads for the attention mechanism.
        """
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim

        # Simplified attention mechanism from paper (Eq. 13)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)

        # For multi-head attention compatibility (optional but good practice)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, observation, node_embeddings, mask):
        """
        Calculates the probability of visiting each node next.

        Args:
            observation (Tensor): The observation vector for the current vehicle,
                                  shape (batch_size, embed_dim).
            node_embeddings (Tensor): Embeddings for all nodes from the encoder,
                                      shape (batch_size, num_nodes, embed_dim).
            mask (Tensor): A boolean mask where True indicates an invalid node,
                           shape (batch_size, num_nodes).

        Returns:
            log_probs (Tensor): Log probabilities for each node, shape (batch_size, num_nodes).
        """
        batch_size, num_nodes, _ = node_embeddings.shape

        # Eq. 13a: Project observation to query
        query = self.W_q(observation) # (batch_size, embed_dim)

        # Eq. 13b: Project node embeddings to keys
        keys = self.W_k(node_embeddings) # (batch_size, num_nodes, embed_dim)

        # Eq. 13c: Calculate compatibility scores (u_i,j,t)
        # Using single-head attention as described in the paper's text
        query = query.unsqueeze(1) # (batch_size, 1, embed_dim)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        scores = scores.squeeze(1) # (batch_size, num_nodes)

        # Apply mask before softmax
        scores[mask] = -float('inf')

        # Eq. 14: Calculate probabilities (softmax) and return log probabilities
        log_probs = torch.log_softmax(scores, dim=-1)

        return log_probs

if __name__ == '__main__':
    # Hyperparameters
    EMBED_DIM = 128
    BATCH_SIZE = 4
    NUM_NODES = 21 # 1 depot + 20 customers

    decoder = Decoder(EMBED_DIM)

    # Dummy inputs
    # Observation vector for the current vehicle
    dummy_observation = torch.rand(BATCH_SIZE, EMBED_DIM)
    # Node embeddings from encoder
    dummy_node_embeddings = torch.rand(BATCH_SIZE, NUM_NODES, EMBED_DIM)
    # Mask: let's say nodes 3 and 5 are invalid for the first batch item
    dummy_mask = torch.zeros(BATCH_SIZE, NUM_NODES, dtype=torch.bool)
    dummy_mask[0, 3] = True
    dummy_mask[0, 5] = True

    log_probs = decoder(dummy_observation, dummy_node_embeddings, dummy_mask)

    print("Decoder Test:")
    print("Observation shape:", dummy_observation.shape)
    print("Node embeddings shape:", dummy_node_embeddings.shape)
    print("Mask shape:", dummy_mask.shape)
    print("Output log_probs shape:", log_probs.shape)
    assert log_probs.shape == (BATCH_SIZE, NUM_NODES)

    # Check if masked probabilities are -inf
    print("Log-prob for masked node [0, 3]:", log_probs[0, 3].item())
    assert log_probs[0, 3].item() == -float('inf')

    # Check if probabilities sum to 1 (or log-probs sum to something reasonable)
    probs = torch.exp(log_probs)
    print("Sum of probabilities for first batch item:", probs[0].sum().item())
    assert math.isclose(probs[0].sum().item(), 1.0, rel_tol=1e-6)

    print("Test passed!")

