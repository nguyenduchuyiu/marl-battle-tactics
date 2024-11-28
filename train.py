import os
from magent2.environments import battle_v4
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import cv2
from tzusunnet import TzuSuNetwork



def blue_policy(observation, agent, epsilon, env, q_network, device):
    if random.random() < epsilon:
        return env.action_space(agent).sample()
    
    observation_tensor = (
        torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
    ).to(device)
    
    with torch.no_grad():
        q_values = q_network(observation_tensor)
        return torch.argmax(q_values).item()

def train_q_network(env, q_network, config):
    """
    Training loop for Q-learning without a replay buffer.

    Args:
        env: The MAgent2 battle environment
        q_network: Neural network for Q-learning
        config: Dictionary containing training hyperparameters
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    q_network = q_network.to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=config["learning_rate"])
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
    
    epsilon = config["epsilon_start"]
    epsilon_decay = (config["epsilon_start"] - config["epsilon_end"]) / config["epsilon_decay_steps"]
    
    episode_rewards = []
    
    # Setup video recording
    vid_dir = "training_videos"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 60
    
    try:
        for episode in range(config["episodes"]):
            env.reset()  
            total_reward = 0
            episode_loss = 0
            frames = []
            
            # For limiting episode length
            time_steps = 0
            
            for agent in env.agent_iter():
                
                time_steps += 1
                if time_steps >= 50000:  # End episode after 50000 time steps
                    print(f"Episode {episode+1} ended at time step {time_steps}")
                    break
                
                observation, reward, termination, truncation, info = env.last()
                done = termination or truncation
                
                if done:
                    action = None  # Agent is dead
                    env.step(action)
                else:
                    agent_handle = agent.split("_")[0]
                    if agent_handle == "blue":
                        # Get action
                        action = blue_policy(observation, agent, epsilon, env, q_network, device)
                        
                        # Convert observation to tensor
                        observation_tensor = (
                            torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                        ).to(device)
                        
                        # Take action
                        env.step(action)
                        
                        # Get next state information
                        next_observation, reward, termination, truncation, info = env.last()
                        next_observation_tensor = (
                            torch.Tensor(next_observation).float().permute([2, 0, 1]).unsqueeze(0)
                        ).to(device)
                        
                        # Calculate Q-values and loss
                        q_values = q_network(observation_tensor)
                        q_value = q_values[0, action]
                        
                        with torch.no_grad():
                            next_q_values = q_network(next_observation_tensor)
                            target = reward + config["gamma"] * next_q_values.max(1)[0] * (1 - float(termination or truncation))

                        
                        loss = F.mse_loss(q_value, target)
                        episode_loss += loss.item()
                        
                        # Optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_reward += reward
                    else:
                        # Random policy for red team
                        action = env.action_space(agent).sample()
                        env.step(action)
                            # Capture frame for video
                            
                if agent == "blue_0":  # Only capture once per full step
                    frames.append(env.render())

                    # Save video for this episode
            if len(frames) > 0:
                height, width, _ = frames[0].shape
                video_path = os.path.join(vid_dir, f"episode_{episode+1}.mp4")
                out = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                for frame in frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()        
                
            # Decay epsilon
            epsilon = max(config["epsilon_end"], epsilon - epsilon_decay)
            
            # Record episode statistics
            episode_rewards.append(total_reward)
            
            print(f"Episode {episode+1}/{config['episodes']} - Total Reward: {total_reward} -Eps Loss: {episode_loss} - Epsilon: {epsilon}")
        
        # Save trained model
        torch.save({
            'model_state_dict': q_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'episode_rewards': episode_rewards
        }, "blue_agent.pt")
        print("Model saved as 'blue_agent.pt'")
        return episode_rewards
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model checkpoint...")
        torch.save({
            'model_state_dict': q_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'episode_rewards': episode_rewards,
            'last_episode': episode
        }, "blue_agent_interrupted.pt")
        print("Model saved as 'blue_agent_interrupted.pt'")
        return episode_rewards

# main

config = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 30,
    "batch_size": 64,
    "episodes": 60,
}

# Initialize environment and network
env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.001,
                        dead_penalty=-5, attack_penalty=-0.01, attack_opponent_reward=1,
                        max_cycles=100, extra_features=False)
env.reset()

q_network = TzuSuNetwork(
    observation_shape=env.observation_space("blue_0").shape,
    action_shape=env.action_space("blue_0").n,
)
    
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
q_network.apply(init_weights)


# Train the network
episode_rewards = train_q_network(env, q_network, config)

