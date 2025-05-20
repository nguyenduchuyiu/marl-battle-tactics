import matplotlib
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

def save_model_checkpoint(model_state_dict, path):
    dir_name = os.path.dirname(path)
    if dir_name: # Ensure directory exists only if path includes a directory
        os.makedirs(dir_name, exist_ok=True)
    torch.save(model_state_dict, path)
    print(f"Model checkpoint saved to {path}")

def linear_epsilon(current_episode_step, eps_start, eps_end, eps_decay_episodes):
    # Epsilon decays based on the number of episodes completed
    return max(eps_end, eps_start - (eps_start - eps_end) * (current_episode_step / eps_decay_episodes))