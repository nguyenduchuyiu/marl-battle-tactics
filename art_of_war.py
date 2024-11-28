from collections import deque, namedtuple
import math
import random
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from magent2.environments import battle_v4
from sun_tzu_network import SunTzuNetwork

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 1
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 5
TAU = 0.01
LR = 0.0001
TARGET_UPDATE_INTERVAL = 1  # Hard update every 10 episodes
GRAD_CLIP = 1000
REPLAY_MEMORY_CAPACITY = 50000
REWARD_SCALE = 1


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def setup_environment():
    return battle_v4.env(map_size=45, step_reward=0,attack_penalty=0, attack_opponent_reward=0.1, max_cycles=1000, render_mode="rgb_array")

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def initialize_networks(observation_shape, action_shape, device, load_path=None):
    policy_net = SunTzuNetwork(observation_shape, action_shape).to(device)
    target_net = SunTzuNetwork(observation_shape, action_shape).to(device)
    
    # Apply weight initialization
    policy_net.apply(initialize_weights)
    target_net.apply(initialize_weights)
    
    if load_path:
        try:
            state_dict = torch.load(load_path, map_location=device, weights_only=True)
            policy_net.load_state_dict(state_dict['policy_net_state_dict'])
            target_net.load_state_dict(state_dict['target_net_state_dict'])
            print(f"Successfully loaded model from {load_path}")
        except FileNotFoundError:
            print(f"Model file not found at {load_path}. Starting with a new model.")
        except KeyError as e:
            print(f"Missing key in state_dict: {e}. Starting with a new model.")
        except Exception as e:
            print(f"Error loading model from {load_path}: {e}. Starting with a new model.")
    else:
        print("No load path provided. Initializing new models.")
    
    return policy_net, target_net

def select_action(policy_net, observation, steps_done, env):
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(observation).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space("blue_0").sample()]], device=device, dtype=torch.long)

def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), GRAD_CLIP)
    optimizer.step()

    return loss.item()

def plot_metrics(episode_rewards, episode_losses, show_result=False):
    plt.figure(1)
    plt.clf()
    if show_result:
        plt.title('Result')
    else:
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Value')

    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    losses_t = torch.tensor(episode_losses, dtype=torch.float)

    plt.plot(rewards_t.numpy(), label='Reward')
    plt.plot(losses_t.numpy(), label='Loss')

    if len(rewards_t) >= 5:
        rewards_means = rewards_t.unfold(0, 5, 1).mean(1).view(-1)
        rewards_means = torch.cat((torch.zeros(4), rewards_means))
        plt.plot(rewards_means.numpy(), label='Reward (mean)')

    if len(losses_t) >= 5:
        losses_means = losses_t.unfold(0, 5, 1).mean(1).view(-1)
        losses_means = torch.cat((torch.zeros(4), losses_means))
        plt.plot(losses_means.numpy(), label='Loss (mean)')

    plt.legend()
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def train(num_episodes, env, policy_net, target_net, optimizer, memory):
    steps_done = 0
    episode_rewards = []
    episode_losses = []

    try:
        for i_episode in range(num_episodes):
            env.reset()
            episode_reward = 0
            running_loss = 0.0
            steps_done += 1

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                done = termination or truncation

                if done:
                    action = None
                    env.step(action)
                else:
                    agent_handle = agent.split("_")[0]
                    if agent_handle == "blue":
                        action = select_action(policy_net, observation, steps_done, env)
                        observation, reward, terminated, truncated, info = env.last()
                        observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                        env.step(action.item())

                        next_observation, reward, termination, truncation, info = env.last()
                        next_observation = torch.tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)
                        reward = torch.tensor([reward * REWARD_SCALE], device=device, dtype=torch.float32)

                        memory.push(observation, action, next_observation, reward)
                        observation = next_observation

                        loss = optimize_model(policy_net, target_net, optimizer, memory)
                        if loss is not None:
                            running_loss += loss

                        if i_episode % TARGET_UPDATE_INTERVAL == 0:
                            target_net.load_state_dict(policy_net.state_dict())

                        target_net_state_dict = target_net.state_dict()
                        policy_net_state_dict = policy_net.state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                        target_net.load_state_dict(target_net_state_dict)
                    else: # red
                        action = env.action_space("red_0").sample()
                        env.step(action)

                episode_reward += reward

            episode_rewards.append(episode_reward)
            episode_losses.append(running_loss)

            print(f'Episode {i_episode + 1}/{num_episodes}')
            print(f'Total Reward: {episode_reward.item():.2f}')
            print(f'Average Loss: {running_loss:.4f}')
            print(f'Epsilon: {max(EPS_END, EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)):.2f}')
            print('-' * 40)

            # plot_metrics(episode_rewards, episode_losses)
        
        save_model(i_episode, policy_net, target_net, optimizer, episode_rewards, episode_losses)
    except KeyboardInterrupt:
        print('Interrupted')
        save_model(i_episode, policy_net, target_net, optimizer, episode_rewards, episode_losses)
    finally:
        print('Complete')
        plot_metrics(episode_rewards, episode_losses, show_result=True)
        plt.ioff()
        plt.show()

def save_model(i_episode, policy_net, target_net, optimizer, episode_rewards, episode_losses):
    torch.save({
        'episode': i_episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
    }, "models/blue.pt")

def main():
    env = setup_environment()
    observation_shape = env.observation_space("blue_0").shape
    action_shape = env.action_space("blue_0").n

    policy_net, target_net = initialize_networks(observation_shape, action_shape, device, load_path='models/blue.pt')
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(REPLAY_MEMORY_CAPACITY)

    train(num_episodes=10, env=env, policy_net=policy_net, target_net=target_net, optimizer=optimizer, memory=memory)

main()