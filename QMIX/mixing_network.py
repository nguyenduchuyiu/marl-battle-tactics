import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim, hypernet_embed=64):
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_shape = state_shape # Kích thước của global state
        self.embed_dim = mixing_embed_dim # Kích thước của embedding layer trong mixing network

        # Hypernetwork để tạo weights cho mixing network layers
        # Layer 1
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_shape, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * self.embed_dim)
        )
        # Bias cho layer 1
        self.hyper_b1 = nn.Linear(state_shape, self.embed_dim)

        # Layer 2
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_shape, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim) # Output là vector embed_dim x 1
        )
        # Bias cho layer 2 (là một scalar)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_shape, self.embed_dim), # Tạm thời để embed_dim, sau đó sẽ qua 1 linear nữa
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        # agent_qs: (batch_size, n_agents) - Q-value của hành động được chọn bởi mỗi agent
        # states: (batch_size, state_shape) - Global state

        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents) # (batch_size, 1, n_agents)

        # Layer 1
        w1 = torch.abs(self.hyper_w1(states)) # (batch_size, n_agents * embed_dim)
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim) # (batch_size, n_agents, embed_dim)
        b1 = self.hyper_b1(states) # (batch_size, embed_dim)
        b1 = b1.view(batch_size, 1, self.embed_dim) # (batch_size, 1, embed_dim)

        # torch.bmm: batch matrix multiplication
        # (batch_size, 1, n_agents) @ (batch_size, n_agents, embed_dim) -> (batch_size, 1, embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1) # (batch_size, 1, embed_dim)

        # Layer 2
        w2 = torch.abs(self.hyper_w2(states)) # (batch_size, embed_dim)
        w2 = w2.view(batch_size, self.embed_dim, 1) # (batch_size, embed_dim, 1)
        b2 = self.hyper_b2(states) # (batch_size, 1)
        b2 = b2.view(batch_size, 1, 1) # (batch_size, 1, 1)

        # (batch_size, 1, embed_dim) @ (batch_size, embed_dim, 1) -> (batch_size, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2 # (batch_size, 1, 1)
        q_total = q_total.view(batch_size, -1) # (batch_size, 1)

        return q_total 