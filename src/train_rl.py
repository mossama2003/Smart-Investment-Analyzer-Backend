"""
train_rl.py

ملف لتدريب Reinforcement Learning (DQN) على الأسهم وصناديق الذهب.
بيتعلم متى يشتري / يبيع / يحتفظ بالـ Asset.
"""

import os
import numpy as np
import pandas as pd
from data_loader import load_all_assets
from features import add_features
from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv
import gymnasium as gym

# ===== الإعدادات =====
MODEL_DIR = "../models/rl/"
os.makedirs(MODEL_DIR, exist_ok=True)
asset_files = ["AAPL.csv", "GOLD_ETF.csv", "EGX30.csv"]
timesteps = 10

# ===== Simple Trading Environment =====
class TradingEnv(gym.Env):
    """
    State = Features آخر N أيام
    Action = 0: Hold, 1: Buy, 2: Sell
    Reward = Profit / Loss
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.features = [col for col in df.columns if col not in ['Date','Close','Open','High','Low','Volume']]
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)
        self.balance = 10000
        self.shares_held = 0

    def reset(self):
        self.current_step = timesteps
        self.balance = 10000
        self.shares_held = 0
        return self._next_observation()

    def _next_observation(self):
        return self.df[self.features].iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        reward = 0
        if action == 1:  # Buy
            self.shares_held += self.balance / current_price
            self.balance = 0
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            reward = self.balance  # reward = current portfolio value

        self.current_step += 1
        done = self.current_step >= len(self.df)-1
        obs = self._next_observation() if not done else np.zeros(len(self.features), dtype=np.float32)
        return obs, reward, done, False, {}

# ===== Load + Features + Train =====
all_assets = load_all_assets(asset_files)

for asset_name, df in all_assets.items():
    print(f"Training RL for {asset_name}...")
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)  # زود العدد لو عندك GPU

    # Save Model
    model_path = os.path.join(MODEL_DIR, f"{asset_name}_rl.zip")
    model.save(model_path)
    print(f"Saved RL model to {model_path}\n")