# Warehouse Robot Reinforcement Learning

This project implements a reinforcement learning (RL) agent to control a robot in a warehouse environment. The agent learns to navigate a grid, avoid obstacles, pick up items, and deliver them to target locations using Proximal Policy Optimization (PPO). test

## Features

- Custom Gymnasium environment for warehouse robot navigation
- PPO agent training and evaluation using Stable Baselines3
- Visualization of robot movement (PyGame window or terminal grid)
- Configurable environment parameters (grid size, obstacles, steps)
- TensorBoard logging for training analysis

## Project Structure

```
.
├── config.py                # Configuration for environment and training
├── env/
│   └── warehouse_env.py     # Custom Gymnasium environment
├── evaluate.py              # Script to evaluate a trained agent
├── train.py                 # Script to train the agent
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── .gitignore               # Git ignore rules
├── ppo_warehouse_robot.zip  # (Generated) Trained model checkpoint
└── ppo_warehouse_tensorboard/
    └── PPO_x/               # (Generated) TensorBoard logs
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd warehouse-robot-rl
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) For graphical visualization:**
   ```bash
   pip install pygame
   ```

## Usage

### Training the Agent

To train the PPO agent from scratch:
```bash
python3 train.py
```
- Model checkpoints will be saved to `ppo_warehouse_robot.zip`.
- Training logs are saved in `ppo_warehouse_tensorboard/`.

### Evaluating the Agent

To evaluate a trained agent and visualize its behavior:
```bash
python3 evaluate.py
```
- Visualization mode is set in `config.py` (`RENDER_MODE = "pygame"` for graphical, `"human"` for text).
- If PyGame is not installed, falls back to text-based rendering.

### Configuration

Edit `config.py` to adjust:
- `GRID_SIZE`: Size of the warehouse grid
- `MAX_STEPS`: Max steps per episode
- `N_OBSTACLES`: Number of obstacles
- `RENDER_MODE`: Visualization mode (`"pygame"` or `"human"`)

### File Descriptions

- `env/warehouse_env.py`: Defines the warehouse environment, robot logic, and rendering.
- `train.py`: Trains the PPO agent.
- `evaluate.py`: Runs the trained agent and visualizes its actions.
- `utils.py`: Helper functions (e.g., printing episode info).
- `requirements.txt`: Lists required Python packages.

## Requirements

- Python 3.8+
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PyGame](https://www.pygame.org/) (optional, for graphical rendering)
- NumPy

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Troubleshooting

- **No graphical window appears:**  
  Ensure PyGame is installed and `RENDER_MODE` is set to `"pygame"` in `config.py`.
- **Text-based grid only:**  
  This is the fallback if PyGame is not installed or if `RENDER_MODE` is `"human"`.
- **Model not found:**  
  Train the agent first using `python3 train.py`.

## License

MIT License
