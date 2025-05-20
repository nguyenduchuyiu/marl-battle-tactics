import torch

class Config:
    def __init__(self):
        # --- MAgent2 Environment Configuration (can be shared or adapted from IDQ) ---
        self.ENV_CONFIG = {
            "map_size": 45, # Example, adjust as needed
            "minimap_mode": False,
            "step_reward": 0.001, # QMIX often uses only the global reward at the end or for specific events
            "dead_penalty": -5, # Example
            "attack_penalty": -0.1, # Example
            "attack_opponent_reward": 1.0, # Example
            "max_cycles": 400, # This will be our episode_limit
            "extra_features": False, # Keep it simple for now
            "render_mode": "rgb_array", # "human" for visualization, "rgb_array" for training
            # For battle_v4, you might want to specify n_friends (blue agents)

            # "n_friends": 8, # Example, will be determined from env
        }

        # --- QMIX Hyperparameters ---
        self.GAMMA = 0.99
        self.LR_AGENT = 0.0005
        self.OPTIMIZER_EPS = 1e-8
        self.RNN_HIDDEN_DIM = 32
        self.QMIX_MIXING_EMBED_DIM = 32
        self.QMIX_HYPERNET_EMBED_DIM = 32
        self.BATCH_SIZE_EPISODES = 8
        self.GRAD_NORM_CLIP = 10
        self.TARGET_UPDATE_INTERVAL_EPISODES = 50
        self.MAX_BUFFER_SIZE_EPISODES = 200
        
        # --- Training Loop ---
        self.NUM_EPISODES_TOTAL = 1000
        self.MIN_EPISODES_FOR_TRAINING = 1
        self.TRAIN_INTERVAL_EPISODES = 1
        self.MODEL_SAVE_INTERVAL_EPISODES = 50
        self.LOG_INTERVAL_EPISODES = 10

        # --- Epsilon-greedy Exploration ---
        self.EPS_START = 1.0
        self.EPS_END = 0.05
        self.EPS_DECAY_EPISODES = 1000

        # --- Red Team Epsilon (if different) ---
        self.RED_EPS_START = 1.0
        self.RED_EPS_END = 0.05
        self.RED_EPS_DECAY_EPISODES = 1000

        # --- Elo Configuration ---
        self.DEFAULT_ELO = 1200
        self.ELO_K_FACTOR = 32
        self.OPPONENT_POOL_MAX_SIZE = 20
        self.OPPONENT_SELECTION_STRATEGY = "random_weighted"
        self.OPPONENT_SWITCH_INTERVAL_EPISODES = 1

        # --- Miscellaneous ---
        self.SEED = 42
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.MODEL_SAVE_DIR = "QMIX/models"
        self.LOG_DIR = "QMIX/logs"

        self.TRAIN_TEAM_PREFIX = "blue"
        self.OPPONENT_TEAM_PREFIX = "red"
