import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from q_network import QNetwork
from utils import linear_epsilon, save_model_checkpoint # Use renamed save_model_checkpoint

class DQNAgent:
    def __init__(self, observation_shape, action_shape, env_action_space_sample_fn,
                 device, lr, tau, gamma, eps_start, eps_end, eps_decay):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.env_action_space_sample_fn = env_action_space_sample_fn
        self.device = device
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay # This is eps_decay_episodes

        self.policy_net = QNetwork(observation_shape, action_shape).to(self.device)
        self.target_net = QNetwork(observation_shape, action_shape).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        
    def select_action(self, observation_np, current_episode_for_epsilon):
        # observation_np is a NumPy array (H,W,C)
        epsilon = linear_epsilon(current_episode_for_epsilon, self.eps_start, self.eps_end, self.eps_decay)
        
        if random.random() < epsilon:
            return self.env_action_space_sample_fn() # Exploration action
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(observation_np, dtype=torch.float32).to(self.device)
                # QNetwork's forward handles single observation (adds batch dim)
                q_values = self.policy_net(obs_tensor) # Output shape (action_shape,)
                return torch.argmax(q_values).cpu().item() # Get action index as Python int

    def optimize_model(self, batch):
        if batch is None:
            return None 

        state_batch = torch.from_numpy(batch['obs']).float().to(self.device)
        action_batch = torch.from_numpy(batch['action']).long().to(self.device) # (B,)
        reward_batch = torch.from_numpy(batch['reward']).float().to(self.device) # (B,)
        next_state_batch = torch.from_numpy(batch['next_obs']).float().to(self.device)
        done_batch = torch.from_numpy(batch['done']).float().to(self.device) # (B,), 1.0 for done, 0.0 for not done

        # action_batch needs to be (B, 1) for gather
        action_batch_reshaped = action_batch.unsqueeze(1)

        # Get Q(s,a) for current states and actions from policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch_reshaped) # (B,1)

        # Get max_a' Q_target(s',a') for next states
        # Initialize with zeros; for terminal states, Q_target(s_terminal,.) = 0
        next_state_values = torch.zeros(state_batch.shape[0], device=self.device) # (B,)
        
        non_final_mask = (done_batch == 0) # True for states that are not done
        non_final_next_states = next_state_batch[non_final_mask]

        if non_final_next_states.size(0) > 0: # If there are any non-final next states
            with torch.no_grad(): # Important: target network computations don't need gradients
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # Compute expected Q values: R + gamma * max_a' Q_target(s',a') (or R if s' is terminal)
        expected_state_action_values = reward_batch + (self.gamma * next_state_values) # (B,)
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) # Target needs to be (B,1)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Gradient clipping
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + \
                                         target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def save_checkpoint(self, path):
        save_model_checkpoint(self.policy_net.state_dict(), path) # Use the utility

    def load_checkpoint(self, path):
        print(f"Loading model checkpoint from {path}")
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_net.train() # Ensure policy_net is in train mode if further training