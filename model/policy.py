import torch
import torch.nn as nn
from model.encoder import Encoder
from model.recorder import RouteRecorder
from model.decoder import Decoder

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_vehicles):
        """
        The main policy network, combining Encoder, Recorders, and Decoder.

        Args:
            input_dim (int): Dimension of the raw input features for each node.
            embed_dim (int): The embedding dimension.
            num_heads (int): Number of attention heads in the encoder.
            num_layers (int): Number of attention layers in the encoder.
            num_vehicles (int): The number of vehicles.
        """
        super(PolicyNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.num_vehicles = num_vehicles

        # --- Main Components ---
        self.encoder = Encoder(input_dim, embed_dim, num_heads, num_layers)
        self.decoder = Decoder(embed_dim, num_heads)

        # --- Route Recorders ---
        # Local recorders for each vehicle
        local_recorder_input_dim = 3 # pos(2) + load(1)
        self.local_recorders = nn.ModuleList(
            [RouteRecorder(local_recorder_input_dim, embed_dim) for _ in range(num_vehicles)]
        )

        # Global recorder
        global_recorder_input_dim = local_recorder_input_dim * num_vehicles
        self.global_recorder = RouteRecorder(global_recorder_input_dim, embed_dim)

    def forward(self, problem_features, vehicle_states, recorder_hidden_states, mask):
        """
        Performs one decoding step for all vehicles.
        Note: In the paper, vehicles are decoded sequentially within a step.
              This implementation simplifies it to a parallel step for efficiency,
              but a sequential loop is needed at the training/inference level.

        Args:
            problem_features (Tensor): Features of all nodes, shape (batch_size, num_nodes, input_dim).
            vehicle_states (Tensor): Current states of all vehicles (pos, load),
                                     shape (batch_size, num_vehicles, 3).
            recorder_hidden_states (tuple): A tuple containing (local_h, global_h).
                                            local_h shape: (batch_size, num_vehicles, embed_dim)
                                            global_h shape: (batch_size, embed_dim)
            mask (Tensor): Mask of invalid nodes for each vehicle,
                           shape (batch_size, num_vehicles, num_nodes).

        Returns:
            log_probs (Tensor): Log probabilities for each vehicle,
                                shape (batch_size, num_vehicles, num_nodes).
            next_hidden_states (tuple): Updated hidden states (next_local_h, next_global_h).
        """
        batch_size, num_nodes, _ = problem_features.shape
        local_h, global_h = recorder_hidden_states

        # --- Encoder ---
        # This would typically be done only once per problem instance
        node_embeddings, graph_embedding = self.encoder(problem_features)

        # --- Route Recorders Update ---
        next_local_h = []
        for i in range(self.num_vehicles):
            h_prev = local_h[:, i, :]
            vehicle_state = vehicle_states[:, i, :]
            h_next = self.local_recorders[i](vehicle_state, h_prev)
            next_local_h.append(h_next.unsqueeze(1))

        next_local_h = torch.cat(next_local_h, dim=1)

        all_vehicle_states = vehicle_states.view(batch_size, -1)
        next_global_h = self.global_recorder(all_vehicle_states, global_h)

        # --- Decoder ---
        log_probs = []
        for i in range(self.num_vehicles):
            # Eq. 12: Construct observation vector
            observation = graph_embedding + next_local_h[:, i, :] + next_global_h

            vehicle_mask = mask[:, i, :]

            # Get log probabilities for the current vehicle
            lp = self.decoder(observation, node_embeddings, vehicle_mask)
            log_probs.append(lp.unsqueeze(1))

        log_probs = torch.cat(log_probs, dim=1)

        return log_probs, (next_local_h, next_global_h)

if __name__ == '__main__':
    # Hyperparameters
    INPUT_DIM = 6 # loc(2)+demand(1)+serv_time(1)+tw(2)
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 3
    NUM_VEHICLES = 3
    BATCH_SIZE = 4
    NUM_NODES = 21

    policy_net = PolicyNetwork(INPUT_DIM, EMBED_DIM, NUM_HEADS, NUM_LAYERS, NUM_VEHICLES)

    # Dummy inputs
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

    print("Policy Network Test:")
    print("Input features shape:", dummy_features.shape)
    print("Input vehicle states shape:", dummy_vehicle_states.shape)
    print("Output log_probs shape:", log_probs.shape)
    print("Next local hidden states shape:", next_local_h.shape)
    print("Next global hidden state shape:", next_global_h.shape)

    assert log_probs.shape == (BATCH_SIZE, NUM_VEHICLES, NUM_NODES)
    assert next_local_h.shape == (BATCH_SIZE, NUM_VEHICLES, EMBED_DIM)
    assert next_global_h.shape == (BATCH_SIZE, EMBED_DIM)
    print("Test passed!")

