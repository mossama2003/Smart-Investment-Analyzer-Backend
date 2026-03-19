# src/train_rl.py

"""
train_rl.py

ملف لتدريب Reinforcement Learning (DQN) على الأسهم وصناديق الذهب.
بيتعلم متى يشتري / يبيع / يحتفظ بالـ Asset.
"""

import os
import numpy as np
from data_loader import load_all_assets
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# ===== تحديد مسار المشروع =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "rl")
os.makedirs(MODEL_DIR, exist_ok=True)

asset_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]
timesteps = 10


# ===== Trading Environment =====
class TradingEnv(gym.Env):
    """
    Simple trading environment.

    State  = Features
    Action = 0: Hold, 1: Buy, 2: Sell
    Reward = Change in portfolio value
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df):
        super().__init__()

        self.df = df.reset_index(drop=True)

        self.features = [
            col for col in df.columns
            if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
        ]

        # Spaces
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.features),),
            dtype=np.float32
        )

        # State
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = timesteps
        self.balance = 10000
        self.shares_held = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return self.df[self.features].iloc[self.current_step].values.astype(np.float32)

    def _get_portfolio_value(self, price):
        return self.balance + self.shares_held * price

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]

        prev_value = self._get_portfolio_value(current_price)

        # ===== تنفيذ الأكشن =====
        if action == 1:  # Buy
            if self.balance > 0:
                self.shares_held = self.balance / current_price
                self.balance = 0

        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance = self.shares_held * current_price
                self.shares_held = 0

        # ===== الانتقال =====
        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        next_price = self.df['Close'].iloc[self.current_step]
        current_value = self._get_portfolio_value(next_price)

        reward = current_value - prev_value

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(len(self.features), dtype=np.float32)
        )

        return obs, reward, terminated, truncated, {}


# ===== Training =====
all_assets = load_all_assets(asset_files)

for asset_name, df in all_assets.items():
    print(f"Training RL for {asset_name}...")

    env = DummyVecEnv([lambda: TradingEnv(df)])

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=100,
        batch_size=32,
    )

    model.learn(total_timesteps=5000)

    # ===== Save =====
    model_path = os.path.join(MODEL_DIR, f"{asset_name}_rl.zip")
    model.save(model_path)

    print(f"✅ Saved RL model to {model_path}\n")