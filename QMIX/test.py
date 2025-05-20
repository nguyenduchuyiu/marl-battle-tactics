from magent2.environments import battle_v4
import torch
from agent_network import RNNAgent
import numpy as np
from config import Config
import torch.nn.functional as F
import random

config = Config()
env = battle_v4.parallel_env(map_size=45, max_cycles=1000, render_mode='human')
obs_shape = env.observation_space("blue_0").shape
blue_agent = RNNAgent(input_shape=obs_shape, n_actions=21, rnn_hidden_dim=config.RNN_HIDDEN_DIM)

blue_agent.load_state_dict(torch.load('/home/huy/Project/marl-battle-tactics/QMIX/models/qmix_blue_ep210_agent.pt',weights_only=True))
red_agent = RNNAgent(input_shape=obs_shape, n_actions=21, rnn_hidden_dim=config.RNN_HIDDEN_DIM)
red_agent.load_state_dict(torch.load('/home/huy/Project/marl-battle-tactics/QMIX/models/qmix_blue_ep210_agent.pt', weights_only=True))
# --- Test Loop for QMIX ---

n_episodes = 5
max_steps = 1000
epsilon = 0.05  # 5% random actions

for ep in range(n_episodes):
    obs = env.reset()
    done = False
    step = 0
    ep_reward_blue = 0
    ep_reward_red = 0

    # Initialize hidden states for all agents
    blue_h = [blue_agent.init_hidden() for _ in range(len(env.agents))]
    red_h = [red_agent.init_hidden() for _ in range(len(env.agents))]

    while not done and step < max_steps:
        blue_actions = []
        red_actions = []

        # Each agent acts
        for i, agent_id in enumerate(env.agents):
            agent_obs = obs[agent_id]
            if "blue" in agent_id:
                # blue_obs = np.fliplr(agent_obs).copy()
                q, blue_h[i] = blue_agent(torch.tensor(agent_obs, dtype=torch.float32), blue_h[i])
                if random.random() < epsilon:
                    action = random.randint(0, q.shape[-1] - 1)
                else:
                    action = q.argmax().item()
                blue_actions.append((agent_id, action))
            else:
                q, red_h[i] = red_agent(torch.tensor(agent_obs, dtype=torch.float32), red_h[i])
                if random.random() < epsilon:
                    action = random.randint(0, q.shape[-1] - 1)
                else:
                    action = q.argmax().item()
                red_actions.append((agent_id, action))

        # Combine actions into a dict
        actions = {aid: a for (aid, a) in red_actions + blue_actions}

        # Step environment
        obs, rewards, dones, truncated, infos = env.step(actions)

        # Sum rewards
        for agent_id, r in rewards.items():
            if "blue" in agent_id:
                ep_reward_blue += r
            else:
                ep_reward_red += r

        # Check if all agents are done
        done = all(dones.values()) or all(truncated.values())
        step += 1

    print(f"Episode {ep+1}: Blue reward = {ep_reward_blue}, Red reward = {ep_reward_red}")










