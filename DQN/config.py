import torch

# --- MAgent2 Environment Configuration ---
ENV_CONFIG = {
    "map_size": 45,
    "minimap_mode": False,
    "step_reward": 0.01,
    "dead_penalty": -2.0,
    "attack_penalty": -0.1,
    "attack_opponent_reward": 2,
    "max_cycles": 100,
    "extra_features": False,
    "render_mode": "rgb_array"
}

# --- Training Hyperparameters ---
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 50  # Interpreted as number of episodes for decay based on notebook logic
TAU = 0.005
LR = 1e-4
NUM_EPISODES = 4  # As per notebook example, for a quick run
REPLAY_BUFFER_CAPACITY = 10000
SEED = 42

# --- Paths ---
MODEL_SAVE_DIR = "models"

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Agent Team Prefixes (used for parsing agent handles) ---
BLUE_AGENT_TEAM_PREFIX = "blue"
RED_AGENT_TEAM_PREFIX = "red"