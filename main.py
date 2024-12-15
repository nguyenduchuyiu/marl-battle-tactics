import random
from magent2.environments import battle_v4
import os
import cv2
from torch_model import QNetwork
import torch


import torch
import torch.nn as nn

class HuyNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        if len(x.shape) == 3:
            batchsize = 1
            x = x.unsqueeze(0)
        else:
            batchsize = x.shape[0]
        x = torch.fliplr(x).permute(0,3,1,2) # flip left-right because blue agent observe identically with red agent
        x = self.cnn(x)
        x = x.reshape(batchsize, -1)
        return self.network(x)



if __name__ == "__main__":
    env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=0,
                        dead_penalty=-1, attack_penalty=0, attack_opponent_reward=1,
                        max_cycles=300, extra_features=False, render_mode="human")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 60
    frames = []
    env.reset()

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def policy(observation, q_network):
        sample = random.random()
        if sample < 0.1:
            return env.action_space("red_0").sample()
        else:
            observation = torch.Tensor(observation).to(device)
            with torch.no_grad():
                q_values = q_network(observation)
            return torch.argmax(q_values, dim=1).cpu().numpy()[0]
        


    
    # Load QNetwork onto the device
    q_networkhuy = HuyNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    ).to(device)

    pretrained_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    ).to(device)    
        
    
    q_networkhuy.load_state_dict(
        torch.load("models/blue_11.pt", map_location=device, weights_only=True)["policy_net_state_dict"]
    )
    

    pretrained_network.load_state_dict(
        torch.load("models/red.pt", map_location=device, weights_only=True)
    )
    
    rewards = 0
    env.reset()
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "blue":
            # if True:
                action =   policy(observation, q_networkhuy)
                # action = policy(observation, q_network)
                # action = blue_policy(observation, sun_tzu_network_refactor)
                # action = env.action_space("red_0").sample()
                rewards += reward
            else:  # red
                # with torch.no_grad():
                
                # action = policy(observation, q_network)
                # action =   policy(observation, pretrained_network)
                # action = env.action_space("red_0").sample()
                observation = (
                    torch.Tensor(observation).float().permute([2,0,1]).unsqueeze(0)
                ).to(device)
                with torch.no_grad():
                    q_value = pretrained_network(observation)
                action = torch.argmax(q_value, dim=1).cpu().numpy()[0]
                # action = torch.randint(1, 21, (1,)).cpu().numpy()[0]

                
                

        env.step(action)

    #     if agent == "blue_0":
    #         frames.append(env.render())

    # height, width, _ = frames[0].shape
    # out = cv2.VideoWriter(
    #     os.path.join(vid_dir, f"huy_vs_nam.mp4"),
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     fps,
    #     (width, height),
    # )
    # for frame in frames:
    #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     out.write(frame_bgr)
    # out.release()
    # print("Done recording")

    env.close()
    print(rewards)
