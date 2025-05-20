import numpy as np
import random
from collections import deque
import torch # Needed for sample_batch if converting to tensors

class EpisodeBuffer:
    def __init__(self, buffer_size_episodes, episode_len_limit, n_agents, obs_shape, state_shape, action_shape, device):
        """
        buffer_size_episodes: Max number of episodes to store.
        episode_len_limit: Max length of an episode (max_cycles from env).
        n_agents: Number of agents in the team we are training.
        obs_shape: Shape of individual agent observation.
        state_shape: Shape of the global state.
        action_shape: Number of actions for an agent (scalar).
        """
        self.buffer_size_episodes = buffer_size_episodes
        self.episode_len_limit = episode_len_limit
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.action_shape = action_shape # This is n_actions
        self.device = device

        # Main storage for completed episodes
        self.episodes_storage = deque(maxlen=self.buffer_size_episodes)
        
        # Temporary buffers for the current episode being built
        self._reset_current_episode_buffers() # Initialize temp buffers
        self.current_episode_step = 0

    def _reset_current_episode_buffers(self):
        
        # Ensure obs_dims and state_dims are tuples for np.zeros
        obs_dims = self.obs_shape if isinstance(self.obs_shape, tuple) else (self.obs_shape,)
        state_dims = self.state_shape if isinstance(self.state_shape, tuple) else (self.state_shape,)

        self.temp_obs_n = np.zeros((self.episode_len_limit, self.n_agents, *obs_dims), dtype=np.float32)
        self.temp_state = np.zeros((self.episode_len_limit, *state_dims), dtype=np.float32)
        self.temp_actions = np.zeros((self.episode_len_limit, self.n_agents, self.action_shape), dtype=np.int64)
        self.temp_reward = np.zeros((self.episode_len_limit, 1), dtype=np.float32)
        self.temp_terminated = np.zeros((self.episode_len_limit, 1), dtype=np.bool_)
        self.temp_filled_mask = np.zeros((self.episode_len_limit, 1), dtype=np.bool_) # For QMIX, indicates valid steps

    def start_new_episode(self):
        if self.current_episode_step > 0:
            # Commit the previously built episode
            episode_data = {
                'obs_n': self.temp_obs_n[:self.current_episode_step].copy(),
                'state': self.temp_state[:self.current_episode_step].copy(),
                'actions': self.temp_actions[:self.current_episode_step].copy(),
                'rewards': self.temp_reward[:self.current_episode_step].copy(),
                'terminated': self.temp_terminated[:self.current_episode_step].copy(),
                'filled_mask': self.temp_filled_mask[:self.current_episode_step].copy(),
                'actual_length': self.current_episode_step # Useful for padding later
            }
            self.episodes_storage.append(episode_data)
        else:
            print("[BUFFER LIFECYCLE] No steps in previous episode, nothing committed.")
        self._reset_current_episode_buffers()
        self.current_episode_step = 0

    def add_transition(self, obs_n, state, actions, reward, terminated):
        if self.current_episode_step >= self.episode_len_limit:
            return

        try:
            self.temp_obs_n[self.current_episode_step] = obs_n
            self.temp_state[self.current_episode_step] = state
            # Ensure actions is (n_agents, action_shape) which is (n_agents, 1)
            self.temp_actions[self.current_episode_step] = actions.reshape(self.n_agents, self.action_shape)
            self.temp_reward[self.current_episode_step] = reward
            self.temp_terminated[self.current_episode_step] = terminated
            self.temp_filled_mask[self.current_episode_step] = True
        except ValueError as e:
            raise e # Re-raise to stop execution and see the error

        self.current_episode_step += 1

    def __len__(self):
        num_stored_episodes = len(self.episodes_storage)
        return num_stored_episodes

    def sample_batch(self, batch_size_episodes):
        if len(self.episodes_storage) < batch_size_episodes:
            return None # Or raise an error

        # Sample episode dictionaries
        sampled_episode_dicts = random.sample(list(self.episodes_storage), batch_size_episodes)

        # --- CRITICAL: Convert list of episode dicts to a batch of tensors ---
        # This part is complex and specific to QMIX. It involves:
        # 1. Finding the max actual_length among sampled_episode_dicts.
        # 2. Padding all episodes' data (obs, state, actions, etc.) to this max_length.
        # 3. Stacking them into tensors.
        # The following is a conceptual placeholder and needs full implementation.

        max_len_in_batch = 0
        for ep_dict in sampled_episode_dicts:
            if ep_dict['actual_length'] > max_len_in_batch:
                max_len_in_batch = ep_dict['actual_length']
        
        # Initialize lists to hold data for each field before converting to tensor
        batch_obs_n_list = []
        batch_state_list = []
        batch_actions_list = []
        batch_reward_list = []
        batch_terminated_list = []
        batch_filled_mask_list = []

        for ep_dict in sampled_episode_dicts:
            ep_len = ep_dict['actual_length']
            
            # Pad obs_n
            padded_obs_n = np.zeros((max_len_in_batch, self.n_agents, *self.temp_obs_n.shape[2:]), dtype=np.float32)
            padded_obs_n[:ep_len] = ep_dict['obs_n']
            batch_obs_n_list.append(padded_obs_n)

            # Pad state
            padded_state = np.zeros((max_len_in_batch, *self.temp_state.shape[1:]), dtype=np.float32)
            padded_state[:ep_len] = ep_dict['state']
            batch_state_list.append(padded_state)
            
            # Pad actions
            padded_actions = np.zeros((max_len_in_batch, self.n_agents, self.action_shape), dtype=np.int64)
            padded_actions[:ep_len] = ep_dict['actions']
            batch_actions_list.append(padded_actions)

            # Pad reward
            padded_reward = np.zeros((max_len_in_batch, 1), dtype=np.float32)
            padded_reward[:ep_len] = ep_dict['rewards']
            batch_reward_list.append(padded_reward)
            
            # Pad terminated
            padded_terminated = np.ones((max_len_in_batch, 1), dtype=np.bool_) # Usually pad with True for terminated
            padded_terminated[:ep_len] = ep_dict['terminated']
            batch_terminated_list.append(padded_terminated)

            # Pad filled_mask
            padded_filled_mask = np.zeros((max_len_in_batch, 1), dtype=np.bool_)
            padded_filled_mask[:ep_len] = ep_dict['filled_mask'] # or np.ones((ep_len,1))
            batch_filled_mask_list.append(padded_filled_mask)

        # Convert lists to tensors
        batch = {
            'obs': torch.tensor(np.array(batch_obs_n_list), dtype=torch.float32).to(self.device),
            'state': torch.tensor(np.array(batch_state_list), dtype=torch.float32).to(self.device),
            'actions': torch.tensor(np.array(batch_actions_list), dtype=torch.long).to(self.device),
            'rewards': torch.tensor(np.array(batch_reward_list), dtype=torch.float32).to(self.device),
            'terminated': torch.tensor(np.array(batch_terminated_list), dtype=torch.bool).to(self.device),
            'filled_mask': torch.tensor(np.array(batch_filled_mask_list), dtype=torch.bool).to(self.device)
        }
        return batch 