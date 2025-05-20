import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, n_actions, rnn_hidden_dim):
        super(RNNAgent, self).__init__()
        self.n_actions = n_actions
        self.rnn_hidden_dim = rnn_hidden_dim

        # Giả sử input_shape là kích thước của output từ CNN (nếu có)
        # Hoặc kích thước của observation nếu không dùng CNN trước RNN
        # Ví dụ: nếu observation là (H, W, C) và bạn có CNN xử lý nó trước,
        # input_shape ở đây sẽ là số features sau khi flatten output của CNN.
        # Nếu không, input_shape là số features của observation đã được flatten.
        # Trong MAgent, observation thường là (map_size, map_size, num_feature_planes)
        # Chúng ta cần một CNN để xử lý nó trước.
        # Dựa trên IDQ/q_network.py, chúng ta có thể tái sử dụng phần CNN.

        # Phần CNN (tương tự IDQ/q_network.py)
        # input_shape ở đây là observation_shape (H, W, C)
        num_channels = input_shape[-1]
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), # Thêm padding để giữ kích thước
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Tính toán flatten_dim sau CNN
        _dummy_cnn_input_shape_chw = (input_shape[2], input_shape[0], input_shape[1])
        dummy_cnn_input_with_batch = torch.randn(1, *_dummy_cnn_input_shape_chw)
        dummy_cnn_output = self.cnn(dummy_cnn_input_with_batch)
        self.cnn_output_features = dummy_cnn_output.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.cnn_output_features, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self):
        # Tạo hidden state khởi tạo (batch_size, rnn_hidden_dim)
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        # obs: (batch_size, H, W, C) hoặc (H, W, C)
        is_single_obs = False
        if obs.dim() == 3: # (H, W, C)
            is_single_obs = True
            obs = obs.unsqueeze(0) # (1, H, W, C)

        # Xử lý CNN
        # Input cho CNN cần là (B, C, H, W)
        # obs hiện tại là (B, H, W, C)
        # 1. torch.fliplr(obs): Flips the last dimension (C). Output: (B, H, W, C_flipped)
        # obs_temp_flipped_channels = torch.fliplr(obs) # Có thể không cần flip kênh cho QMIX
        # 2. .permute(0, 3, 1, 2): Original (B,H,W,C) -> New (B, C, H, W)
        obs_processed_for_cnn = obs.permute(0, 3, 1, 2)

        cnn_out = self.cnn(obs_processed_for_cnn) # (B, C_out, H_out, W_out)
        x_flattened = cnn_out.reshape(cnn_out.shape[0], -1) # (B, cnn_output_features)

        x = F.relu(self.fc1(x_flattened))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h) # (B, n_actions)

        if is_single_obs:
            return q.squeeze(0), h.squeeze(0) # (n_actions,), (rnn_hidden_dim,)
        return q, h
