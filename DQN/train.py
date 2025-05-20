import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# Assuming MAgent2 is installed and accessible
from magent2.environments import battle_v4

from replay_buffer import MultiAgentReplayBuffer
from dqn_agent import DQNAgent
from utils import linear_epsilon
import config # Import all constants from config.py

def set_seeds(seed_value):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    # For full reproducibility, you might also consider:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def main():
    set_seeds(config.SEED)

    print(f"Using device: {config.DEVICE}")

    env = battle_v4.env(**config.ENV_CONFIG)
    # Initial reset, seed for reproducibility
    # Note: MAgent2's env.reset() might not take a seed argument directly in all versions.
    # If it doesn't, seeding is primarily via global np/random/torch seeds.
    try:
        env.reset(seed=config.SEED) 
    except TypeError:
        print("Warning: env.reset() does not accept a seed argument. Global seeds will be used.")
        env.reset()


    # Get observation and action space details
    # Assuming "blue_0" is representative for blue team and "red_0" for red team
    try:
        obs_shape_blue = env.observation_space(config.BLUE_AGENT_TEAM_PREFIX + "_0").shape
        action_shape_blue = env.action_space(config.BLUE_AGENT_TEAM_PREFIX + "_0").n
        action_shape_red = env.action_space(config.RED_AGENT_TEAM_PREFIX + "_0").n # For red agent's random actions
    except Exception as e:
        print(f"Error accessing observation/action space (e.g., for 'blue_0' or 'red_0'): {e}")
        print("Ensure MAgent2 environment is correctly initialized and agent names are valid.")
        print("Using default shapes as a fallback (may be incorrect for your specific MAgent2 version):")
        obs_shape_blue = (13, 13, 5) # Example fallback
        action_shape_blue = 21       # Example fallback
        action_shape_red = 21        # Example fallback
        # return # Or exit if critical info is missing

    print(f"Observation shape (blue): {obs_shape_blue}, Action shape (blue): {action_shape_blue}")

    # Exploration strategy for blue agent (matches notebook: uses red's action space for sampling)
    blue_explore_action_fn = lambda: env.action_space(config.RED_AGENT_TEAM_PREFIX + "_0").sample()
    
    blue_agent = DQNAgent(
        observation_shape=obs_shape_blue,
        action_shape=action_shape_blue,
        env_action_space_sample_fn=blue_explore_action_fn,
        device=config.DEVICE,
        lr=config.LR, tau=config.TAU, gamma=config.GAMMA,
        eps_start=config.EPS_START, eps_end=config.EPS_END, eps_decay=config.EPS_DECAY
    )

    buffer = MultiAgentReplayBuffer(
        capacity=config.REPLAY_BUFFER_CAPACITY,
        observation_shape=obs_shape_blue, # Store observations of blue agents
        action_shape=1 # Action is a single integer index
    )

    episode_rewards_history = []
    episode_losses_history = []
    # This counter is incremented per episode for epsilon decay, as in the notebook
    completed_episodes_counter = 0 

    # Ensure model save directory exists
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    for i_episode in range(config.NUM_EPISODES):
        try:
            env.reset(seed=config.SEED + i_episode)
        except TypeError:
            env.reset() # If seed per episode reset is not supported

        current_episode_total_reward = 0.0
        current_episode_total_loss = 0.0
        num_optim_steps_this_episode = 0
        
        completed_episodes_counter += 1 # For epsilon decay

        # MAgent2's agent_iter loop
        for agent_handle in env.agent_iter():
            observation, reward_for_last_action, termination, truncation, info = env.last()
            agent_is_done = termination or truncation

            # Parse agent team and index (e.g., "blue_0" -> team "blue", index "0")
            try:
                team_name, agent_idx_str = agent_handle.split("_")
            except ValueError:
                print(f"Warning: Could not parse agent_handle '{agent_handle}'. Skipping.")
                if agent_is_done: env.step(None) # Step with None if done to advance iter
                continue


            if team_name == config.BLUE_AGENT_TEAM_PREFIX:
                # The reward from env.last() is for this blue agent's *previous* action.
                # Update the reward for that stored transition.
                buffer.update_last_reward(agent_idx_str, reward_for_last_action)
                current_episode_total_reward += reward_for_last_action

                if agent_is_done:
                    env.step(None) # Agent is done, cannot act. Step to advance iterator.
                    continue

                # Agent is alive and it's its turn to act
                action_to_take = blue_agent.select_action(observation, completed_episodes_counter)
                env.step(action_to_take)

                # Get next state for (observation, action_to_take)
                try:
                    next_obs_for_buffer = env.observe(agent_handle) # Observe this agent after its action
                    next_obs_terminal_flag = False
                except Exception: # Agent might have died due to its action
                    # Use a placeholder for next_observation if agent is now dead
                    next_obs_for_buffer = np.zeros(obs_shape_blue, dtype=np.float32) 
                    next_obs_terminal_flag = True
                
                # Reward for (observation, action_to_take) is initially unknown (placeholder 0.0).
                # It will be updated when this agent (agent_idx_str) is selected again.
                reward_placeholder = 0.0
                buffer.push(agent_idx_str, observation, action_to_take, 
                              reward_placeholder, next_obs_for_buffer, next_obs_terminal_flag)
                
                # Optimize model
                batch_data = buffer.sample(config.BATCH_SIZE)
                if batch_data:
                    loss = blue_agent.optimize_model(batch_data)
                    if loss is not None:
                        current_episode_total_loss += loss
                        num_optim_steps_this_episode += 1
                
                blue_agent.update_target_network() # Soft update target network

            elif team_name == config.RED_AGENT_TEAM_PREFIX:
                if agent_is_done:
                    env.step(None)
                    continue
                # Red agent acts randomly
                red_random_action = np.random.randint(0, action_shape_red)
                env.step(red_random_action)
            
            else: # Unknown agent team
                if agent_is_done: env.step(None)
                else: env.step(None) # Or some default action / error


        # End of episode
        episode_rewards_history.append(current_episode_total_reward)
        avg_loss_this_episode = current_episode_total_loss / num_optim_steps_this_episode if num_optim_steps_this_episode > 0 else 0.0
        episode_losses_history.append(avg_loss_this_episode)

        current_epsilon_val = linear_epsilon(completed_episodes_counter, config.EPS_START, config.EPS_END, config.EPS_DECAY)
        print(f"Episode {i_episode + 1}/{config.NUM_EPISODES} completed.")
        print(f"  Total Reward: {current_episode_total_reward:.2f}")
        print(f"  Average Loss: {avg_loss_this_episode:.4f}")
        print(f"  Epsilon: {current_epsilon_val:.3f}")
        print("-" * 40)
        
        # --- Periodically print average reward ---
        if (i_episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards_history[-10:])
            print(f"[Periodic] Average reward for last 10 episodes (up to episode {i_episode + 1}): {avg_reward:.2f}")

        # Save model periodically and at the end
        if (i_episode + 1) % 10 == 0 or (i_episode + 1) == config.NUM_EPISODES:
            path_to_save = os.path.join(config.MODEL_SAVE_DIR, f"blue_dqn_episode_{i_episode+1}.pt")
            blue_agent.save_checkpoint(path_to_save)

    # Final actions after training loop
    final_model_path = os.path.join(config.MODEL_SAVE_DIR, "blue_dqn_final.pt")
    blue_agent.save_checkpoint(final_model_path)
    
    # --- Plotting training curves ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episode_losses_history, label="Average Loss", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Episode Losses")
    plt.legend()

    plt.tight_layout()
    plt.show()

    env.close()
    print("Training finished.")

if __name__ == "__main__":
    main()