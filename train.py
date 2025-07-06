from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from env.warehouse_env import WarehouseRobotEnv
import config as cfg
import os

# Prepare tensorboard dir
os.makedirs(cfg.TENSORBOARD_LOG, exist_ok=True)

# ✅ Add render_mode=None for training
env = WarehouseRobotEnv(
    grid_size=cfg.GRID_SIZE,
    max_steps=cfg.MAX_STEPS,
    n_obstacles=cfg.N_OBSTACLES,
    render_mode=None
)
check_env(env, warn=True)

vec_env = DummyVecEnv([
    lambda: WarehouseRobotEnv(
        grid_size=cfg.GRID_SIZE,
        max_steps=cfg.MAX_STEPS,
        n_obstacles=cfg.N_OBSTACLES,
        render_mode=None  # Prevent PyGame windows during training
    )
])

# Train
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=cfg.TENSORBOARD_LOG)
model.learn(total_timesteps=cfg.TOTAL_TIMESTEPS)
model.save(cfg.MODEL_PATH)

# Notify
print(f"✅ Training complete and model saved to '{cfg.MODEL_PATH}.zip'")
