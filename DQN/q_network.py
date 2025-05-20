import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape): # observation_shape: (H, W, C)
        super().__init__()
        num_channels = observation_shape[-1]

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3), # Stride 1, Padding 0 by default
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3), # Stride 1, Padding 0 by default
            nn.ReLU(),
        )
        
        # Calculate flatten_dim dynamically
        # Create a dummy input with batch dimension: (1, C, H, W)
        # observation_shape is (H,W,C), so permute to (C,H,W) for CNN's expected input channel order
        _dummy_cnn_input_shape_chw = (observation_shape[2], observation_shape[0], observation_shape[1]) 
        dummy_cnn_input_with_batch = torch.randn(1, *_dummy_cnn_input_shape_chw)
        
        dummy_cnn_output = self.cnn(dummy_cnn_input_with_batch) # Output shape (1, num_channels, H_out, W_out)
        flatten_dim = dummy_cnn_output.view(1, -1).shape[1] # Total features after CNN

        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        # x input: (B, H, W, C) or (H, W, C)
        assert len(x.shape) >= 3, "Input tensor must have at least 3 dimensions (H, W, C) or (B, H, W, C)"
        
        is_single_obs = False
        if len(x.shape) == 3: # Input is (H, W, C)
            is_single_obs = True
            x = x.unsqueeze(0) # Add batch dimension: (1, H, W, C)

        batch_size = x.shape[0]
        
        # Transformations as per the notebook:
        # x is (B, H, W, C)
        # 1. torch.fliplr(x): Flips the last dimension (C). Output: (B, H, W, C_flipped)
        x_temp_flipped_channels = torch.fliplr(x)
        # 2. .permute(0, 3, 1, 2): Original (B,H,W,C_flipped) -> New (B, C_flipped, H, W)
        x_processed_for_cnn = x_temp_flipped_channels.permute(0, 3, 1, 2)
        
        x_cnn_out = self.cnn(x_processed_for_cnn) # CNN processes (B, C_flipped, H, W)
        x_flattened = x_cnn_out.reshape(batch_size, -1) # Reshape to (B, Features)
        q_values = self.network(x_flattened)
        
        if is_single_obs:
            return q_values.squeeze(0) # Remove batch dim if input was single: (action_shape,)
        return q_values # (B, action_shape)