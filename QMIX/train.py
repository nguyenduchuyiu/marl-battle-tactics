import torch.nn as nn
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

# Assuming MAgent2 is installed and accessible
from magent2.environments import battle_v4

from replay_buffer import EpisodeBuffer
from qmix_learner import QMIXLearner
from utils import linear_epsilon, set_seeds, get_agent_ids, obs_list_to_state_vector
from config import Config # Import all constants from QMIX/config.py
from agent_network import RNNAgent # Import RNNAgent to create opponent network instances
from elo_manager import EloManager, update_elo_ratings, DEFAULT_ELO

def main():
    import pandas as pd
    import torch
    import numpy as np
    import random
    import os
    import matplotlib.pyplot as plt

    # Assuming MAgent2 is installed and accessible
    from magent2.environments import battle_v4


    config = Config()
    set_seeds(config.SEED)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print(f"Using device: {config.DEVICE}")
    if config.DEVICE == "cuda":
        print(f"Number of CUDA devices available: {torch.cuda.device_count()}")

    print(f"Training team (Blue): {config.TRAIN_TEAM_PREFIX}")
    print(f"Opponent team (Red): {config.OPPONENT_TEAM_PREFIX}")

    # --- Initialize Environment ---
    env_config_copy = config.ENV_CONFIG.copy()
    env = battle_v4.parallel_env(**env_config_copy)
    try:
        env.reset(seed=config.SEED)
    except TypeError:
        print("Warning: env.reset() does not accept a seed argument. Global seeds will be used.")
        env.reset()

    # --- Get Agent and Environment Details ---
    blue_team_agent_ids = sorted([agent_id for agent_id in env.agents if agent_id.startswith(config.TRAIN_TEAM_PREFIX)])
    red_team_agent_ids = sorted([agent_id for agent_id in env.agents if agent_id.startswith(config.OPPONENT_TEAM_PREFIX)])

    if not blue_team_agent_ids:
        raise ValueError(f"No agents found for Blue team with prefix: {config.TRAIN_TEAM_PREFIX}")
    if not red_team_agent_ids:
        raise ValueError(f"No agents found for Red team with prefix: {config.OPPONENT_TEAM_PREFIX}")

    n_agents_blue_team = len(blue_team_agent_ids)
    n_agents_red_team = len(red_team_agent_ids)
    print(f"Number of agents in Blue team: {n_agents_blue_team}")
    print(f"Number of agents in Red team: {n_agents_red_team}")

    first_blue_agent_id = blue_team_agent_ids[0]
    obs_shape_blue_agent = env.observation_space(first_blue_agent_id).shape
    action_shape_blue_agent = env.action_space(first_blue_agent_id).n

    first_red_agent_id = red_team_agent_ids[0]
    obs_shape_red_agent = env.observation_space(first_red_agent_id).shape
    action_shape_red_agent = env.action_space(first_red_agent_id).n

    global_state_shape_tuple = env.state().shape
    global_state_shape_flat = int(np.prod(global_state_shape_tuple))

    print(f"Blue Agent Observation Shape: {obs_shape_blue_agent}")
    print(f"Blue Agent Action Shape (Number of actions): {action_shape_blue_agent}")
    print(f"Red Agent Observation Shape: {obs_shape_red_agent}")
    print(f"Red Agent Action Shape: {action_shape_red_agent}")
    print(f"Global State Shape (raw tuple for buffer): {global_state_shape_tuple}")
    print(f"Global State Shape (flattened for mixer): {global_state_shape_flat}")

    # --- Initialize Learners and Buffers for Both Teams ---
    print("Initializing a QMIX learner for the Blue team.")
    blue_learner = QMIXLearner(
        n_agents=n_agents_blue_team,
        obs_shape=obs_shape_blue_agent,
        state_shape=global_state_shape_flat,
        n_actions=action_shape_blue_agent,
        args=config
    )

    episode_buffer_blue = EpisodeBuffer(
        buffer_size_episodes=config.MAX_BUFFER_SIZE_EPISODES,
        episode_len_limit=config.ENV_CONFIG["max_cycles"],
        n_agents=n_agents_blue_team,
        obs_shape=obs_shape_blue_agent,
        state_shape=global_state_shape_tuple,
        action_shape=1,
        device=config.DEVICE
    )

    episode_rewards_history_blue = []
    episode_rewards_history_red = []
    episode_losses_history = []
    blue_elo_history = []

    total_steps_counter = 0
    completed_episodes_counter = 0

    # --- Start try block for training ---
    try:
        print(f"Starting training for {config.NUM_EPISODES_TOTAL} episodes with a learner for the Blue team...")

        elo_manager_registry_file = os.path.join(config.MODEL_SAVE_DIR, "elo_pool_registry.json")
        elo_manager_models_dir = os.path.join(config.MODEL_SAVE_DIR, "elo_opponent_models_storage")

        elo_manager = EloManager(
            pool_registry_file=elo_manager_registry_file,
            max_pool_size=config.OPPONENT_POOL_MAX_SIZE,
            opponent_models_storage_dir=elo_manager_models_dir
        )
        current_blue_team_elo = config.DEFAULT_ELO

        # Prepare opponent agent net for loading policies
        opponent_agent_net = RNNAgent(
            input_shape=obs_shape_red_agent,
            n_actions=action_shape_red_agent,
            rnn_hidden_dim=config.RNN_HIDDEN_DIM
        ).to(config.DEVICE)

        opponent_using_data_parallel = False # Flag to track if opponent_agent_net is using DataParallel
        if config.DEVICE == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using DataParallel for Opponent RNNAgent across {torch.cuda.device_count()} GPUs.")
            opponent_agent_net = nn.DataParallel(opponent_agent_net)
            opponent_using_data_parallel = True
        opponent_agent_net.eval()

        # Initialize hidden state for Red team's opponent network (will be updated when policy loads)
        if opponent_using_data_parallel:
            current_red_team_hidden_states = opponent_agent_net.module.init_hidden().squeeze(0).repeat(n_agents_red_team, 1).cpu().numpy()
        else:
            current_red_team_hidden_states = opponent_agent_net.init_hidden().squeeze(0).repeat(n_agents_red_team, 1).cpu().numpy()    
        for i_episode in range(config.NUM_EPISODES_TOTAL):
            print(f"======== Episode {i_episode+1} of {config.NUM_EPISODES_TOTAL} =========")
            episode_buffer_blue.start_new_episode()
            obs_dict = env.reset()

            if blue_learner.using_data_parallel:
                current_blue_team_hidden_states = blue_learner.policy_agent_net.module.init_hidden().squeeze(0).repeat(n_agents_blue_team, 1).cpu().numpy()
            else:
                current_blue_team_hidden_states = blue_learner.policy_agent_net.init_hidden().squeeze(0).repeat(n_agents_blue_team, 1).cpu().numpy()
            
            # current_red_team_hidden_states will be re-initialized below if a new opponent is loaded

            current_episode_total_reward_blue = 0
            current_episode_total_reward_red = 0
            current_episode_steps = 0
            terminated_episode = False
            current_red_team_opponent_obj = None

            # Opponent selection for Red team
            needs_new_opponent = (i_episode == 0) or \
                                    (i_episode > 0 and i_episode % config.OPPONENT_SWITCH_INTERVAL_EPISODES == 0) or \
                                    (current_red_team_opponent_obj is None and len(elo_manager.opponent_pool) > 0)

            if needs_new_opponent:
                selected_opponent_obj = elo_manager.select_opponent(
                    current_agent_elo=current_blue_team_elo,
                    strategy=config.OPPONENT_SELECTION_STRATEGY
                )
                if selected_opponent_obj:
                    print(f"Red team loading opponent with Elo: {selected_opponent_obj.elo}")
                    try:
                        # state_dict_from_file is clean (no "module." prefix)
                        state_dict_from_file = torch.load(selected_opponent_obj.policy_path, map_location=config.DEVICE)
                        if opponent_using_data_parallel:
                            opponent_agent_net.module.load_state_dict(state_dict_from_file)
                        else:
                            opponent_agent_net.load_state_dict(state_dict_from_file)
                        current_red_team_opponent_obj = selected_opponent_obj
                    except Exception as e:
                        print(f"Error loading opponent policy {selected_opponent_obj.policy_path}: {e}. Red using Blue's current policy as fallback.")
                        if blue_learner.using_data_parallel:
                            state_dict_to_load = blue_learner.policy_agent_net.module.state_dict()
                        else:
                            state_dict_to_load = blue_learner.policy_agent_net.state_dict()
                        
                        # Load the clean state_dict_to_load appropriately
                        if opponent_using_data_parallel:
                            opponent_agent_net.module.load_state_dict(state_dict_to_load)
                        else:
                            opponent_agent_net.load_state_dict(state_dict_to_load)
                        current_red_team_opponent_obj = None # Fallback is not an Elo pool opponent
                else:
                    print("No opponent selected from Elo pool or pool empty. Red team uses copy of Blue's current policy.")
                    if blue_learner.using_data_parallel:
                        state_dict_to_load = blue_learner.policy_agent_net.module.state_dict()
                    else:
                        state_dict_to_load = blue_learner.policy_agent_net.state_dict()
                    
                    # Load the clean state_dict_to_load appropriately
                    if opponent_using_data_parallel:
                        opponent_agent_net.module.load_state_dict(state_dict_to_load)
                    else:
                        opponent_agent_net.load_state_dict(state_dict_to_load)
                    current_red_team_opponent_obj = None

                # IMPORTANT: Re-initialize hidden state for the newly loaded policy in opponent_agent_net
                if opponent_using_data_parallel:
                    current_red_team_hidden_states = opponent_agent_net.module.init_hidden().squeeze(0).repeat(n_agents_red_team, 1).cpu().numpy()
                else:
                    current_red_team_hidden_states = opponent_agent_net.init_hidden().squeeze(0).repeat(n_agents_red_team, 1).cpu().numpy()
                opponent_agent_net.eval()

            for step_in_episode in range(config.ENV_CONFIG["max_cycles"]):
                if config.ENV_CONFIG.get("render_mode") == "human":
                    env.render()

                # Get current global state (s_t) *before* taking an action
                current_global_state_s_t = env.state()

                # --- Action Selection for Blue Team ---
                blue_team_obs_list = [obs_dict.get(agent_id, np.zeros(obs_shape_blue_agent, dtype=np.float32)) for agent_id in blue_team_agent_ids]
                blue_team_obs = [np.fliplr(obs).copy() for obs in blue_team_obs_list] # Flip the observation horizontally for blue agents
                blue_team_obs_np = np.array(blue_team_obs)
                q_vals_blue_team, next_hidden_states_blue_team_np = blue_learner.get_policy_agent_q_values(
                    blue_team_obs_np, current_blue_team_hidden_states
                )
                epsilon_blue = linear_epsilon(completed_episodes_counter, config.EPS_START, config.EPS_END, config.EPS_DECAY_EPISODES)
                blue_team_actions = [
                    random.randint(0, action_shape_blue_agent - 1) if random.random() < epsilon_blue else np.argmax(q_vals_blue_team[i])
                    for i in range(n_agents_blue_team)
                ]

                # --- Action Selection for Red Team ---
                red_team_obs_list = [obs_dict.get(agent_id, np.zeros(obs_shape_red_agent, dtype=np.float32)) for agent_id in red_team_agent_ids]
                red_team_obs_np = np.array(red_team_obs_list)

                # Convert to tensors for opponent_agent_net
                red_team_obs_tensor = torch.tensor(red_team_obs_np, dtype=torch.float32).to(config.DEVICE)
                red_team_hidden_tensor = torch.tensor(current_red_team_hidden_states, dtype=torch.float32).to(config.DEVICE)

                # Get Q-values from opponent_agent_net (which has the loaded policy)
                with torch.no_grad():
                    q_vals_red_team_tensor, next_hidden_states_red_team_tensor = opponent_agent_net(
                        red_team_obs_tensor, red_team_hidden_tensor
                    )
                q_vals_red_team_numpy = q_vals_red_team_tensor.cpu().numpy()
                next_hidden_states_red_team_np = next_hidden_states_red_team_tensor.cpu().numpy()

                epsilon_red = linear_epsilon(
                    completed_episodes_counter,
                    config.RED_EPS_START,
                    config.RED_EPS_END,
                    config.RED_EPS_DECAY_EPISODES
                )
                red_team_actions = [
                    random.randint(0, action_shape_red_agent - 1) if random.random() < epsilon_red else np.argmax(q_vals_red_team_numpy[i])
                    for i in range(n_agents_red_team)
                ]

                # --- Combine actions and Step Environment ---
                actions_to_env = {}
                for i, agent_id in enumerate(blue_team_agent_ids):
                    if agent_id in obs_dict: actions_to_env[agent_id] = blue_team_actions[i]
                for i, agent_id in enumerate(red_team_agent_ids):
                    if agent_id in obs_dict: actions_to_env[agent_id] = red_team_actions[i]

                next_obs_dict, rewards_dict, dones_dict, truncs_dict, infos_dict = env.step(actions_to_env)

                global_reward_blue = sum(rewards_dict.get(agent_id, 0) for agent_id in blue_team_agent_ids)
                global_reward_red = sum(rewards_dict.get(agent_id, 0) for agent_id in red_team_agent_ids)

                current_episode_total_reward_blue += global_reward_blue
                current_episode_total_reward_red += global_reward_red

                # terminated_episode_flag is True if s_{t+1} is a true terminal state
                terminated_episode_flag = dones_dict.get("__all__", False)
                # truncated_episode_flag is True if s_{t+1} is a truncated state (e.g., time limit)
                truncated_episode_flag = truncs_dict.get("__all__", False)
                
                # Episode ends if it's a true termination OR a truncation
                episode_actually_over = terminated_episode_flag or truncated_episode_flag

                # --- Store Transition in Buffer for Blue Team ---
                # obs_n: o_t (observations at current step t)
                # state: s_t (global state at current step t)
                # actions: u_t (actions taken at current step t)
                # reward: r_t (reward received after u_t)
                # terminated: True if s_{t+1} (next state) is a true terminal state (not truncated)
                blue_actions_np_reshaped = np.array(blue_team_actions).reshape(n_agents_blue_team, 1)
                episode_buffer_blue.add_transition(
                    obs_n=blue_team_obs_np,
                    state=current_global_state_s_t, # Pass s_t
                    actions=blue_actions_np_reshaped,
                    reward=global_reward_blue,
                    terminated=terminated_episode_flag # Pass only true termination flag
                )

                # Store Red team's transition in the same buffer (if needed, adjust for s_t and terminated flag)
                # Assuming red_team_obs_np is o_t for red, and we need s_t (which is current_global_state_s_t)
                # and terminated_episode_flag applies to the whole environment state.
                red_actions_np_reshaped = np.array(red_team_actions).reshape(n_agents_red_team, 1)
                episode_buffer_blue.add_transition(
                    obs_n=red_team_obs_np, # o_t for red
                    state=current_global_state_s_t, # s_t (same global state)
                    actions=red_actions_np_reshaped, # u_t for red
                    reward=global_reward_red, # r_t for red
                    terminated=terminated_episode_flag # True if s_{t+1} is terminal
                )

                obs_dict = next_obs_dict
                current_blue_team_hidden_states = next_hidden_states_blue_team_np
                current_red_team_hidden_states = next_hidden_states_red_team_np

                total_steps_counter += 1
                current_episode_steps += 1

                if episode_actually_over:
                    break

            # --- End of Episode ---
            completed_episodes_counter += 1
            episode_rewards_history_blue.append(current_episode_total_reward_blue)
            episode_rewards_history_red.append(current_episode_total_reward_red)
            blue_elo_history.append(current_blue_team_elo)

            # --- Learning Update ---
            if completed_episodes_counter % config.TRAIN_INTERVAL_EPISODES == 0:
                if completed_episodes_counter >= config.MIN_EPISODES_FOR_TRAINING:
                    can_train_blue = len(episode_buffer_blue) >= config.BATCH_SIZE_EPISODES

                    if can_train_blue:
                        print(f"Training at episode {completed_episodes_counter}")
                        batch_blue = episode_buffer_blue.sample_batch(config.BATCH_SIZE_EPISODES)
                        if batch_blue is not None and batch_blue["obs"].shape[1] > 0:
                            loss_b = blue_learner.train(batch_blue, completed_episodes_counter)
                            if loss_b is not None:
                                episode_losses_history.append(loss_b)
                            print(f"Completed training for Blue team at episode {completed_episodes_counter}")
                        else:
                            print(f"Skipping training for Blue team at episode {completed_episodes_counter}: batch invalid or empty.")
                    else:
                        print(f"Skipping training at episode {completed_episodes_counter}: Blue buffer not ready ({len(episode_buffer_blue)}/{config.BATCH_SIZE_EPISODES}).")
                else:
                    print(f"Skipping training at episode {completed_episodes_counter}: min episodes ({config.MIN_EPISODES_FOR_TRAINING}) not reached.")

            # --- Logging ---
            if (i_episode + 1) % config.LOG_INTERVAL_EPISODES == 0:
                avg_reward_blue = np.mean(episode_rewards_history_blue[-config.LOG_INTERVAL_EPISODES:]) if episode_rewards_history_blue else 0
                avg_loss_blue = episode_losses_history[-1] if episode_losses_history else 0
                avg_reward_red = np.mean(episode_rewards_history_red[-config.LOG_INTERVAL_EPISODES:]) if episode_rewards_history_red else 0

                print(f"Episode: {i_episode+1}/{config.NUM_EPISODES_TOTAL} | Steps: {total_steps_counter}")
                print(f"  Blue Team - Avg Reward: {avg_reward_blue:.2f}, Epsilon: {epsilon_blue:.2f}")
                print(f"  Red Team  - Avg Reward: {avg_reward_red:.2f}, Epsilon: {epsilon_red:.2f}")
                print(f"  Blue Learner - Avg Loss: {avg_loss_blue:.4f}")

            # --- Model Saving and Elo Pool Update ---
            if (i_episode + 1) % config.MODEL_SAVE_INTERVAL_EPISODES == 0:
                model_path_prefix = os.path.join(config.MODEL_SAVE_DIR, f"qmix_blue_ep{i_episode+1}")
                blue_learner.save_models(model_path_prefix)
                print(f"Saved Blue learner model components at {model_path_prefix}")

                # Path to the agent model file specifically for the Elo pool
                agent_model_path_for_elo = f"{model_path_prefix}_agent.pt"

                if os.path.exists(agent_model_path_for_elo):
                    snapshot_name = f"blue_ep{i_episode+1}_elo{current_blue_team_elo:.0f}"
                    elo_manager.add_opponent_to_pool(
                        source_policy_path=agent_model_path_for_elo,
                        initial_elo=current_blue_team_elo,
                        name_prefix=snapshot_name
                    )
                else:
                    print(f"Warning: Agent model {agent_model_path_for_elo} not found after saving. Cannot add to Elo pool.")

            # --- Elo Update ---
            if current_red_team_opponent_obj:
                score_for_blue = 0.5
                if current_episode_total_reward_blue > current_episode_total_reward_red:
                    score_for_blue = 1.0
                elif current_episode_total_reward_red > current_episode_total_reward_blue:
                    score_for_blue = 0.0

                opponent_elo_before_match = current_red_team_opponent_obj.elo
                new_blue_elo, new_opponent_elo = update_elo_ratings(
                    current_blue_team_elo,
                    opponent_elo_before_match,
                    score_for_blue,
                    k_factor=config.ELO_K_FACTOR
                )

                print(f"Elo Update after Ep {i_episode+1}: Blue ({current_blue_team_elo:.0f}->{new_blue_elo:.0f}) vs "
                        f"Red ({opponent_elo_before_match:.0f}->{new_opponent_elo:.0f}). "
                        f"Blue score: {score_for_blue}. Blue Reward: {current_episode_total_reward_blue:.2f}, Red Reward: {current_episode_total_reward_red:.2f}")

                current_blue_team_elo = new_blue_elo
                elo_manager.update_opponent_stats(current_red_team_opponent_obj.name, new_opponent_elo)
            else:
                print(f"Ep {i_episode+1}: Blue Reward: {current_episode_total_reward_blue:.2f}, Red Reward (non-Elo opponent): {current_episode_total_reward_red:.2f}. No Elo update.")
            
            print(f"=====================================================")

        # --- End of Training ---
        env.close()
        print("Training finished.")

    except Exception as e:
        print(f"Exception occurred during training: {e}")

    finally:
        # --- Plotting ---
        plt.figure(figsize=(16, 6))

        # --- Plot Rewards ---
        plt.subplot(1, 3, 1)
        episodes = np.arange(1, len(episode_rewards_history_blue) + 1)
        plt.plot(episodes, episode_rewards_history_blue, label="Blue Team Reward", color='blue', alpha=0.4)
        plt.plot(episodes, episode_rewards_history_red, label="Red Team Reward", color='red', alpha=0.4)

        # Moving average for rewards
        window = min(50, len(episodes))
        if window > 1:
            blue_ma = pd.Series(episode_rewards_history_blue).rolling(window, min_periods=1).mean()
            red_ma = pd.Series(episode_rewards_history_red).rolling(window, min_periods=1).mean()
            plt.plot(episodes, blue_ma, label=f"Blue Reward MA({window})", color='blue', linewidth=2)
            plt.plot(episodes, red_ma, label=f"Red Reward MA({window})", color='red', linewidth=2)

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode Rewards (with Moving Average)")
        plt.legend()
        plt.grid(True)

        # --- Plot Loss ---
        plt.subplot(1, 3, 2)
        if episode_losses_history:
            steps = np.arange(1, len(episode_losses_history) + 1)
            plt.plot(steps, episode_losses_history, label="Blue Learner Loss", color='purple', alpha=0.4)
            # Moving average for loss
            window_loss = min(50, len(steps))
            if window_loss > 1:
                loss_ma = pd.Series(episode_losses_history).rolling(window_loss, min_periods=1).mean()
                plt.plot(steps, loss_ma, label=f"Loss MA({window_loss})", color='purple', linewidth=2)
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training Loss (with Moving Average)")
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, "No loss data", ha='center', va='center')
            plt.axis('off')

        # --- Plot Blue Team Elo ---
        plt.subplot(1, 3, 3)
        if blue_elo_history:
            plt.plot(episodes, blue_elo_history, label="Blue Team Elo", color='orange')
            # Moving average for Elo
            window_elo = min(50, len(episodes))
            if window_elo > 1:
                elo_ma = pd.Series(blue_elo_history).rolling(window_elo, min_periods=1).mean()
                plt.plot(episodes, elo_ma, label=f"Elo MA({window_elo})", color='orange', linewidth=2)
            plt.xlabel("Episode")
            plt.ylabel("Elo")
            plt.title("Blue Team Elo (with Moving Average)")
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, "No Elo data", ha='center', va='center')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(config.LOG_DIR, "training_summary.png"))
        plt.show()


if __name__ == '__main__':
    main() 