import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agent_network import RNNAgent
from mixing_network import QMixer


class QMIXLearner:
    def __init__(self, n_agents, obs_shape, state_shape, n_actions, args):
        self.n_agents = n_agents
        self.obs_shape = obs_shape # Shape of individual agent observation
        self.state_shape = state_shape # Shape of global state
        self.n_actions = n_actions
        self.args = args # Dictionary or Namespace containing hyperparameters
        self.device = args.DEVICE

        # Agent networks (policy and target)
        self.policy_agent_net = RNNAgent(obs_shape, n_actions, args.RNN_HIDDEN_DIM).to(self.device)
        self.target_agent_net = RNNAgent(obs_shape, n_actions, args.RNN_HIDDEN_DIM).to(self.device)

        # Mixing networks (policy and target)
        self.policy_mixer_net = QMixer(n_agents, state_shape, args.QMIX_MIXING_EMBED_DIM, args.QMIX_HYPERNET_EMBED_DIM).to(self.device)
        self.target_mixer_net = QMixer(n_agents, state_shape, args.QMIX_MIXING_EMBED_DIM, args.QMIX_HYPERNET_EMBED_DIM).to(self.device)

        # Apply DataParallel if multiple GPUs are available
        self.using_data_parallel = False
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using DataParallel for QMIXLearner across {torch.cuda.device_count()} GPUs.")
            self.policy_agent_net = nn.DataParallel(self.policy_agent_net)
            self.policy_mixer_net = nn.DataParallel(self.policy_mixer_net)
            self.using_data_parallel = True
        
        self.target_agent_net.load_state_dict(self.policy_agent_net.module.state_dict() if self.using_data_parallel else self.policy_agent_net.state_dict())
        self.target_agent_net.eval() # Target network is not trained directly

        self.target_mixer_net.load_state_dict(self.policy_mixer_net.module.state_dict() if self.using_data_parallel else self.policy_mixer_net.state_dict())
        self.target_mixer_net.eval()

        # Optimizer for both agent and mixer networks
        # Parameters are correctly accessed whether DataParallel is used or not
        self.params = list(self.policy_agent_net.parameters()) + list(self.policy_mixer_net.parameters())
        self.optimizer = optim.Adam(params=self.params, lr=args.LR_AGENT, eps=args.OPTIMIZER_EPS)

        self.last_target_update_episode = 0

    def train(self, batch, episode_num):
        """
        Trains the QMIX networks on a batch of episodes.
        batch: Dictionary from EpisodeBuffer.sample()
               Keys: "obs", "state", "actions", "rewards", "terminated", "filled_mask"
               Shapes:
                 obs: (batch_size, max_seq_len, n_agents, *obs_shape)
                 state: (batch_size, max_seq_len, state_shape)
                 actions: (batch_size, max_seq_len, n_agents, 1)
                 rewards: (batch_size, max_seq_len, 1)
                 terminated: (batch_size, max_seq_len, 1)
                 filled_mask: (batch_size, max_seq_len, 1)
        episode_num: Current episode number, for target network updates.
        """
        batch_size = batch["obs"].shape[0]
        max_seq_len = batch["obs"].shape[1]

        # Move batch to device and ensure correct types
        obs_batch = batch["obs"].to(torch.float32).to(self.device)
        state_batch = batch["state"].to(torch.float32).to(self.device)
        actions_batch = batch["actions"].to(torch.int64).to(self.device)      # (bs, seq, N, 1)
        rewards_batch = batch["rewards"].to(torch.float32).to(self.device)    # (bs, seq, 1)
        terminated_batch = batch["terminated"].to(torch.float32).to(self.device) # (bs, seq, 1)
        filled_mask = batch["filled_mask"].to(torch.float32).to(self.device)  # (bs, seq, 1)

        # Flatten state for mixer
        state_batch = state_batch.view(batch_size, max_seq_len, -1)  # (batch, seq, 10125)

        # --- Calculate Q-values for chosen actions using policy networks ---
        # Initialize hidden states for RNN agents
        if self.using_data_parallel:
            policy_hidden_states = self.policy_agent_net.module.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1) # (bs, N, rnn_hidden_dim)
        else:
            policy_hidden_states = self.policy_agent_net.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1) # (bs, N, rnn_hidden_dim)

        chosen_action_qvals = []
        for t in range(max_seq_len):
            # obs_t: (bs, N, *obs_shape)
            obs_t = obs_batch[:, t]
            # Reshape for RNNAgent: (bs * N, *obs_shape)
            obs_t_reshaped = obs_t.reshape(batch_size * self.n_agents, *self.obs_shape)
            policy_hidden_states_reshaped = policy_hidden_states.reshape(batch_size * self.n_agents, -1)

            # agent_q_t: (bs * N, n_actions), policy_hidden_states: (bs * N, rnn_hidden_dim)
            agent_q_t, policy_hidden_states_updated = self.policy_agent_net(obs_t_reshaped, policy_hidden_states_reshaped)
            
            # Reshape hidden_states back: (bs, N, rnn_hidden_dim)
            policy_hidden_states = policy_hidden_states_updated.reshape(batch_size, self.n_agents, -1)

            # Gather Q-values for actions taken: actions_batch[:, t] is (bs, N, 1)
            # agent_q_t needs to be (bs, N, n_actions)
            agent_q_t_reshaped = agent_q_t.reshape(batch_size, self.n_agents, self.n_actions)
            # actions_t: (bs, N, 1)
            actions_t = actions_batch[:, t]
            
            # chosen_q_t: (bs, N)
            chosen_q_t = torch.gather(agent_q_t_reshaped, dim=2, index=actions_t).squeeze(2)
            chosen_action_qvals.append(chosen_q_t)

        # chosen_action_qvals_all_t: (bs, seq_len, N)
        chosen_action_qvals_all_t = torch.stack(chosen_action_qvals, dim=1)

        # --- Calculate Q_tot from policy network by iterating over sequence length ---
        q_total_policy_list = []
        for t in range(max_seq_len):
            current_chosen_qvals = chosen_action_qvals_all_t[:, t, :] # Shape: (bs, N)
            current_state = state_batch[:, t, :]                   # Shape: (bs, state_shape)
            
            # Apply filled_mask for mixer input? Generally, mixer processes all, loss is masked.
            # If a step is padding, its q_total_t won't contribute to loss due to filled_mask later.
            current_state = current_state.reshape(-1, current_state.shape[-1])  # (batch*seq, 10125)
            q_total_t = self.policy_mixer_net(current_chosen_qvals, current_state) # Shape: (bs*seq, 1)
            q_total_policy_list.append(q_total_t)
        
        # q_total_policy: (bs, seq_len, 1)
        q_total_policy = torch.stack(q_total_policy_list, dim=1)
        q_total_policy = q_total_policy.view(batch_size, max_seq_len, 1)  # reshape lại nếu cần

        # --- Calculate Target Q_tot using target networks ---
        # Get Q-values for all actions from target agent network for all timesteps in obs_batch
        target_agent_q_all_steps_list = []
        temp_target_hidden_states = self.target_agent_net.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1) # (bs, N, rnn_hidden_dim)

        for t in range(max_seq_len):
            obs_t_reshaped = obs_batch[:, t].reshape(batch_size * self.n_agents, *self.obs_shape)
            
            target_q_at_t, temp_target_hidden_states_next = self.target_agent_net(
                obs_t_reshaped,
                temp_target_hidden_states.reshape(batch_size * self.n_agents, -1)
            )
            target_q_at_t = target_q_at_t.reshape(batch_size, self.n_agents, self.n_actions) # (bs, N, n_actions)
            temp_target_hidden_states = temp_target_hidden_states_next.reshape(batch_size, self.n_agents, -1)
            target_agent_q_all_steps_list.append(target_q_at_t)

        # target_agent_q_all_steps_stacked: (bs, seq_len, N, n_actions)
        # These are Q_i(o_t, h_t; theta_agent_target)
        target_agent_q_all_steps_stacked = torch.stack(target_agent_q_all_steps_list, dim=1)
        
        # Select max Q-value for each agent at each step: max_{a_i} Q_i(o_t, h_t; theta_agent_target)
        # target_max_qvals_agents: (bs, seq_len, N)
        target_max_qvals_agents = target_agent_q_all_steps_stacked.max(dim=3)[0]

        # --- Calculate Q_tot_target from target mixer by iterating over sequence length ---
        # These are Q_tot(s_t, u'_t; theta_mixer_target)
        q_total_target_for_s_t_list = []
        for t in range(max_seq_len):
            current_target_max_qvals = target_max_qvals_agents[:, t, :] # Shape: (bs, N)
            current_state_for_target = state_batch[:, t, :]           # Shape: (bs, state_shape)
            current_state_for_target = current_state_for_target.reshape(-1, current_state_for_target.shape[-1])  # (batch*seq, 10125)
            q_total_target_s_t = self.target_mixer_net(current_target_max_qvals, current_state_for_target) # Shape: (bs*seq, 1)
            q_total_target_for_s_t_list.append(q_total_target_s_t)
        
        # q_total_target_for_s_t_stacked: (bs, seq_len, 1)
        q_total_target_for_s_t_stacked = torch.stack(q_total_target_for_s_t_list, dim=1)

        # For TD target y_t = r_t + gamma * (1-term_t) * Q_tot_target(s_{t+1}, u'_{t+1})
        # We need to shift q_total_target_for_s_t_stacked to get values for s_{t+1}
        # Q_tot_target_next_state will be [Q_tot(s_1, u'_1), ..., Q_tot(s_T, u'_T)]
        # where Q_tot(s_T, u'_T) is 0 if episode ends at T-1.
        # The values from q_total_target_for_s_t_stacked are Q_tot(s_0, u'_0) ... Q_tot(s_{T-1}, u'_{T-1})
        
        # Get Q_tot_target(s_1, u'_1) ... Q_tot_target(s_{T-1}, u'_{T-1})
        q_total_target_values_for_next_state = q_total_target_for_s_t_stacked[:, 1:, :]
        
        # For the state after the last state in sequence (s_T), Q_tot_target is 0
        # (handled by (1-terminated) if the episode actually ends, or just 0 if it's padding)
        final_step_target_q_val = torch.zeros_like(q_total_target_values_for_next_state[:, 0:1, :]) # (bs, 1, 1)
        
        # Concatenate to get Q_tot_target(s_{t+1}) for t = 0 to max_seq_len-1
        # Resulting shape: (bs, seq_len, 1)
        q_total_target = torch.cat([q_total_target_values_for_next_state, final_step_target_q_val], dim=1)

        # --- Calculate TD Target ---
        # y_i = r_i + gamma * (1 - terminated_i) * Q_tot_target_i
        # rewards_batch: (bs, seq, 1)
        # terminated_batch: (bs, seq, 1)
        # q_total_target: (bs, seq, 1)
        td_targets = rewards_batch + self.args.GAMMA * (1 - terminated_batch) * q_total_target

        # Detach td_targets as they are considered fixed for the loss calculation
        td_targets = td_targets.detach()

        # --- Calculate Loss ---
        # Loss = (Q_tot_policy - td_targets)^2, masked by `filled_mask`
        # q_total_policy: (bs, seq, 1)
        # td_targets: (bs, seq, 1)
        # filled_mask: (bs, seq, 1)
        
        td_error = (q_total_policy - td_targets)
        masked_td_error = td_error * filled_mask

        # Mean squared error, averaged over valid steps
        loss = (masked_td_error ** 2).sum() / filled_mask.sum()
        # --- Optimization ---
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.params, self.args.GRAD_NORM_CLIP)
        self.optimizer.step()

        # --- Update Target Networks (Hard Update) ---
        if self.args.TARGET_UPDATE_INTERVAL_EPISODES > 0 and \
           (episode_num - self.last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL_EPISODES >= 1.0:
            self.update_target_networks()
            self.last_target_update_episode = episode_num
            # print(f"Updated target networks at episode {episode_num}")

        return loss.item()

    def update_target_networks(self):
        if self.using_data_parallel:
            self.target_agent_net.load_state_dict(self.policy_agent_net.module.state_dict())
            self.target_mixer_net.load_state_dict(self.policy_mixer_net.module.state_dict())
        else:
            self.target_agent_net.load_state_dict(self.policy_agent_net.state_dict())
            self.target_mixer_net.load_state_dict(self.policy_mixer_net.state_dict())

    def get_policy_agent_q_values(self, obs_batch, hidden_states_batch):
        """
        Helper to get Q-values from policy agent network for action selection.
        obs_batch: (n_agents_in_env, *obs_shape) - current observations for all agents
        hidden_states_batch: (n_agents_in_env, rnn_hidden_dim) - current hidden states
        Returns:
            q_vals: (n_agents_in_env, n_actions)
            next_hidden_states: (n_agents_in_env, rnn_hidden_dim)
        """
        obs_batch_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
        hidden_states_tensor = torch.tensor(hidden_states_batch, dtype=torch.float32).to(self.device)
        
        # RNNAgent expects (batch_size, *features) for obs and (batch_size, hidden_dim) for hidden
        # Here, n_agents_in_env acts as the batch_size for the agent network
        q_vals, next_hidden_states = self.policy_agent_net(obs_batch_tensor, hidden_states_tensor)
        return q_vals.detach().cpu().numpy(), next_hidden_states.detach().cpu().numpy()

    def save_models(self, path_prefix):
        agent_state_dict = self.policy_agent_net.module.state_dict() if self.using_data_parallel else self.policy_agent_net.state_dict()
        mixer_state_dict = self.policy_mixer_net.module.state_dict() if self.using_data_parallel else self.policy_mixer_net.state_dict()
        
        torch.save(agent_state_dict, f"{path_prefix}_agent.pt")
        torch.save(mixer_state_dict, f"{path_prefix}_mixer.pt")
        print(f"Saved models to {path_prefix}_agent.pt and {path_prefix}_mixer.pt")

    def load_models(self, path_prefix):
        agent_state_dict = torch.load(f"{path_prefix}_agent.pt", map_location=self.device)
        mixer_state_dict = torch.load(f"{path_prefix}_mixer.pt", map_location=self.device)

        if self.using_data_parallel:
            self.policy_agent_net.module.load_state_dict(agent_state_dict)
            self.policy_mixer_net.module.load_state_dict(mixer_state_dict)
        else:
            # If the saved model has "module." prefix (e.g. from an older DataParallel save not using .module)
            # and current model is not DataParallel, it might need stripping.
            # However, our save_models now saves clean state_dicts.
            self.policy_agent_net.load_state_dict(agent_state_dict)
            self.policy_mixer_net.load_state_dict(mixer_state_dict)
            
        self.update_target_networks() # Ensure target networks are also updated
        
        # Set to eval if only for inference, train if for further training
        # policy_agent_net and policy_mixer_net are set to train mode by default
        # and only set to eval mode explicitly when needed (e.g. during action selection if not training)
        # For now, let's assume after loading, we might continue training or use for eval.
        # If strictly for inference after loading, .eval() should be called.
        # self.policy_agent_net.eval()
        # self.policy_mixer_net.eval()
        print(f"Loaded models from {path_prefix}_agent.pt and {path_prefix}_mixer.pt") 