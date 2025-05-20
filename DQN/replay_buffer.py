from collections import defaultdict, deque
import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, capacity, observation_shape, action_shape): # action_shape is not strictly used here
        self.capacity = capacity
        self.observation_shape = observation_shape 
        # self.action_shape = action_shape # Not used in current implementation based on notebook

        # Use a defaultdict to automatically create deques for new agents
        self.buffers = defaultdict(lambda: {
            'obs': deque(maxlen=capacity),
            'action': deque(maxlen=capacity),
            'reward': deque(maxlen=capacity),
            'next_obs': deque(maxlen=capacity),
            'done': deque(maxlen=capacity),
        })

    def push(self, agent_id, obs, action, reward, next_obs, done):
        self.buffers[agent_id]['obs'].append(obs)
        self.buffers[agent_id]['action'].append(action)
        self.buffers[agent_id]['reward'].append(reward)
        self.buffers[agent_id]['next_obs'].append(next_obs)
        self.buffers[agent_id]['done'].append(done)

    def sample(self, batch_size):
        all_agent_ids = list(self.buffers.keys())
        if not all_agent_ids:
            return None  # No agents in the buffer

        # Check if we have enough data to sample
        total_transitions = sum(len(self.buffers[agent_id]['obs']) for agent_id in all_agent_ids)
        if total_transitions < batch_size:
            return None # Not enough samples across all agents

        # Collect transitions from all agents into a single list
        all_transitions_list = []
        for agent_id in all_agent_ids:
            agent_buffer = self.buffers[agent_id]
            for i in range(len(agent_buffer['obs'])):
                all_transitions_list.append({
                    'obs': agent_buffer['obs'][i],
                    'action': agent_buffer['action'][i],
                    'reward': agent_buffer['reward'][i],
                    'next_obs': agent_buffer['next_obs'][i],
                    'done': agent_buffer['done'][i]
                })
        
        if not all_transitions_list: # Should be covered by total_transitions check
            return None

        # Sample indices from the combined transitions
        # Ensure replace=False if total_transitions >= batch_size, else it might error or need replace=True
        can_sample_without_replacement = total_transitions >= batch_size
        indices = np.random.choice(len(all_transitions_list), batch_size, replace=not can_sample_without_replacement)

        # Extract the sampled transitions
        obs_batch = np.array([all_transitions_list[i]['obs'] for i in indices])
        action_batch = np.array([all_transitions_list[i]['action'] for i in indices])
        reward_batch = np.array([all_transitions_list[i]['reward'] for i in indices])
        next_obs_batch = np.array([all_transitions_list[i]['next_obs'] for i in indices])
        done_batch = np.array([all_transitions_list[i]['done'] for i in indices])

        return {
            'obs': obs_batch,
            'action': action_batch,
            'reward': reward_batch,
            'next_obs': next_obs_batch,
            'done': done_batch
        }

    def update_last_reward(self, agent_id, new_reward):
        if agent_id not in self.buffers or not self.buffers[agent_id]['reward']:
            # print(f"Warning: Cannot update reward for agent {agent_id}. Buffer empty or agent not found.")
            return
        self.buffers[agent_id]['reward'][-1] = new_reward

    def __len__(self):
        return sum(len(self.buffers[agent_id]['obs']) for agent_id in self.buffers)

    def clear(self, agent_id=None):
        if agent_id:
            if agent_id in self.buffers:
                self.buffers[agent_id]['obs'].clear()
                self.buffers[agent_id]['action'].clear()
                self.buffers[agent_id]['reward'].clear()
                self.buffers[agent_id]['next_obs'].clear()
                self.buffers[agent_id]['done'].clear()
        else: # Clear all agents
            for aid in list(self.buffers.keys()): # Iterate over copy of keys
                self.clear(aid)
            self.buffers.clear()