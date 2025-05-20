import numpy as np
import torch
import os
import random

def linear_epsilon(current_step, eps_start, eps_end, eps_decay_steps):
    """
    Calculates epsilon for epsilon-greedy exploration using linear decay.
    Decay happens over `eps_decay_steps`.
    """
    fraction = min(1.0, float(current_step) / eps_decay_steps)
    return eps_start + fraction * (eps_end - eps_start)

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

def get_agent_ids(env, team_prefix):
    """Gets a sorted list of agent IDs for a given team prefix."""
    agent_ids = [agent_id for agent_id in env.agents if agent_id.startswith(team_prefix)]
    agent_ids.sort() # Ensure consistent order
    return agent_ids

def get_state_size(n_agents, obs_shape_agent):
    """
    Calculates the size of the global state, assuming it's a concatenation
    of all agent observations (flattened).
    obs_shape_agent: (H, W, C)
    """
    if isinstance(obs_shape_agent, tuple) and len(obs_shape_agent) == 3:
        return n_agents * obs_shape_agent[0] * obs_shape_agent[1] * obs_shape_agent[2]
    elif isinstance(obs_shape_agent, int): # If obs is already flat
        return n_agents * obs_shape_agent
    else:
        raise ValueError(f"Unsupported obs_shape_agent: {obs_shape_agent}")

def obs_list_to_state_vector(obs_list):
    """
    Converts a list of agent observations (each potentially a 3D tensor HWC)
    into a single flattened state vector.
    Assumes obs_list contains numpy arrays.
    """
    flat_obs = [obs.flatten() for obs in obs_list]
    return np.concatenate(flat_obs) 