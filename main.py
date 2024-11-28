import random
from magent2.environments import battle_v4
import os
import cv2
from torch_model import QNetwork
import torch

from sun_tzu_network import SunTzuNetwork



if __name__ == "__main__":
    env = battle_v4.env(map_size=45, step_reward=0, attack_penalty=-1, attack_opponent_reward=0, dead_penalty=-1, max_cycles=200, render_mode="rgb_array")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 60
    frames = []

    # random policies
    # env.reset()
    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()

    #     if termination or truncation:
    #         action = None  # this agent has died
    #     else:
    #         action = env.action_space(agent).sample()

    #     env.step(action)

    #     if agent == "red_0":
    #         frames.append(env.render())

    # height, width, _ = frames[0].shape
    # out = cv2.VideoWriter(
    #     os.path.join(vid_dir, f"random.mp4"),
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     fps,
    #     (width, height),
    # )
    # for frame in frames:
    #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     out.write(frame_bgr)
    # out.release()
    # print("Done recording random agents")

    # pretrained policies
    frames = []
    env.reset()

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def red_policy(observation, q_network):
        sample = random.random()
        if sample < 0.1:
            return env.action_space("red_0").sample()
        else:
            observation = (
                torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                q_values = q_network(observation)
            return torch.argmax(q_values, dim=1).cpu().numpy()[0]
        
    def blue_policy(observation, sun_tzu_network):
        sample = random.random()
        if sample < 0.1:
            return env.action_space("blue_0").sample()
        else:
            observation = (torch.tensor(observation, dtype=torch.float32).to(device).unsqueeze(0))
            with torch.no_grad():
                return sun_tzu_network(observation).max(1).indices.view(1, 1).item()
        
    # Load SunTzuNetwork onto the deviceattr
    sun_tzu_network = SunTzuNetwork(
        env.observation_space("blue_0").shape, env.action_space("blue_0").n
    ).to(device)
    
    sun_tzu_network.load_state_dict(torch.load("models/blue_722.pt", map_location=device, weights_only=True)["policy_net_state_dict"])

    # Load QNetwork onto the device
    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    ).to(device)
    q_network.load_state_dict(
        torch.load("models/red.pt", weights_only=True, map_location=device)
    )
    rewards = 0
    env.reset()
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "red":
            # if True:
                # action = red_policy(observation, q_network)
                action = blue_policy(observation, sun_tzu_network)
            else:  # blue
                action = blue_policy(observation, sun_tzu_network)
                rewards += reward

        env.step(action)

        if agent == "red_0":
            frames.append(env.render())

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"sun-tzu_vs_sun-tzu.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording")

    env.close()
    print(rewards)
