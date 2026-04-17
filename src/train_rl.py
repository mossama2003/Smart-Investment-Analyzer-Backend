# src/train_rl.py

"""
train_rl.py

FIXES applied:
1. All training logic wrapped in train_rl() — no top-level side effects on import.
2. add_features() called before building the environment so state space is consistent
   with what predict.py sends.
"""

import os
import numpy as np
from src.data_loader import load_all_assets
from src.features import add_features
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "rl")

asset_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]
TIMESTEPS_LOOKBACK = 10


class TradingEnv(gym.Env):
    """
    Simple stock trading environment.
    State  = engineered features for the current row
    Action = 0: Hold, 1: Buy, 2: Sell
    Reward = change in portfolio value
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = [
            col for col in df.columns
            if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
        ]
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.features),),
            dtype=np.float32,
        )
        self.current_step = 0
        self.balance = 10_000.0
        self.shares_held = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = TIMESTEPS_LOOKBACK
        self.balance = 10_000.0
        self.shares_held = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        return self.df[self.features].iloc[self.current_step].values.astype(np.float32)

    def _portfolio_value(self, price):
        return self.balance + self.shares_held * price

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        prev_value = self._portfolio_value(current_price)

        if action == 1 and self.balance > 0:
            self.shares_held = self.balance / current_price
            self.balance = 0.0
        elif action == 2 and self.shares_held > 0:
            self.balance = self.shares_held * current_price
            self.shares_held = 0.0

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        next_price = self.df['Close'].iloc[self.current_step]
        reward = self._portfolio_value(next_price) - prev_value

        obs = (
            self._get_obs() if not terminated
            else np.zeros(len(self.features), dtype=np.float32)
        )
        return obs, reward, terminated, truncated, {}


def train_rl():
    os.makedirs(MODEL_DIR, exist_ok=True)
    all_assets = load_all_assets(asset_files)

    for asset_name, df in all_assets.items():
        print(f"Training RL for {asset_name}...")

        # FIX: use add_features so state space matches predict.py
        df = add_features(df)

        env = DummyVecEnv([lambda d=df: TradingEnv(d)])

        model = DQN(
            "MlpPolicy", env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=10_000,
            learning_starts=100,
            batch_size=32,
        )
        model.learn(total_timesteps=5_000)

        model_path = os.path.join(MODEL_DIR, f"{asset_name}_rl.zip")
        model.save(model_path)
        print(f"✅ Saved RL model to {model_path}\n")

    return True


if __name__ == "__main__":
    train_rl()