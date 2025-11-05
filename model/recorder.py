import torch
import torch.nn as nn

class RouteRecorder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        """
        Route Recorder module using a GRU cell.
        This can be used as both a local and global recorder.

        Args:
            input_dim (int): Dimension of the input at each step.
                             For local: vehicle state (e.g., location + load).
                             For global: combined state of all vehicles.
            embed_dim (int): The embedding dimension (hidden state size).
        """
        super(RouteRecorder, self).__init__()
        self.gru_cell = nn.GRUCell(input_dim, embed_dim)

    def forward(self, x, h_prev):
        """
        Forward pass for one step.

        Args:
            x (Tensor): Input tensor for the current step, shape (batch_size, input_dim).
            h_prev (Tensor): Previous hidden state, shape (batch_size, embed_dim).

        Returns:
            h_next (Tensor): Next hidden state, shape (batch_size, embed_dim).
        """
        h_next = self.gru_cell(x, h_prev)
        return h_next

if __name__ == '__main__':
    # Hyperparameters
    EMBED_DIM = 128
    BATCH_SIZE = 4
    NUM_VEHICLES = 3

    # --- Local Route Recorder Test ---
    # Input: current location (2D) + remaining load (1D) = 3
    local_input_dim = 3
    local_recorder = RouteRecorder(local_input_dim, EMBED_DIM)

    # Dummy input for one vehicle
    dummy_vehicle_state = torch.rand(BATCH_SIZE, local_input_dim)
    # Initial hidden state (usually zeros)
    h_prev_local = torch.zeros(BATCH_SIZE, EMBED_DIM)

    h_next_local = local_recorder(dummy_vehicle_state, h_prev_local)

    print("Local Route Recorder Test:")
    print("Input state shape:", dummy_vehicle_state.shape)
    print("Previous hidden state shape:", h_prev_local.shape)
    print("Next hidden state shape:", h_next_local.shape)
    assert h_next_local.shape == (BATCH_SIZE, EMBED_DIM)
    print("Test passed!")
    print("-" * 20)

    # --- Global Route Recorder Test ---
    # Input: Concatenation of all vehicles' states
    global_input_dim = local_input_dim * NUM_VEHICLES
    global_recorder = RouteRecorder(global_input_dim, EMBED_DIM)

    # Dummy input for all vehicles combined
    dummy_all_vehicles_state = torch.rand(BATCH_SIZE, global_input_dim)
    # Initial hidden state
    h_prev_global = torch.zeros(BATCH_SIZE, EMBED_DIM)

    h_next_global = global_recorder(dummy_all_vehicles_state, h_prev_global)

    print("Global Route Recorder Test:")
    print("Input state shape:", dummy_all_vehicles_state.shape)
    print("Previous hidden state shape:", h_prev_global.shape)
    print("Next hidden state shape:", h_next_global.shape)
    assert h_next_global.shape == (BATCH_SIZE, EMBED_DIM)
    print("Test passed!")

