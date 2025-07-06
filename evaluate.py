import time
from stable_baselines3 import PPO
from env.warehouse_env import WarehouseRobotEnv
import config as cfg

# Load trained model
model = PPO.load(cfg.MODEL_PATH)

# Setup environment (must match training)
env = WarehouseRobotEnv(
    grid_size=cfg.GRID_SIZE,
    max_steps=cfg.MAX_STEPS,
    n_obstacles=cfg.N_OBSTACLES,
    render_mode=cfg.RENDER_MODE
)

# Run evaluation episodes
for ep in range(100):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    print(f"\n🎮 Episode {ep + 1} Start")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.1)  # Slow down rendering

    print(f"🏁 Episode {ep + 1} finished | Total Reward: {total_reward}")

env.close()
