import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class WarehouseRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "pygame"]}

    def __init__(self, grid_size=10, max_steps=100, n_obstacles=10, render_mode="human"):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_obstacles = n_obstacles
        self.render_mode = render_mode
        self.window = None
        self.cell_size = 60
        
        # Enhanced oscillation detection
        self.position_history = []
        self.history_limit = 15  # Increased to catch longer cycles
        self.recent_positions = []  # Track last 6 positions for cycle detection
        self.recent_limit = 6
        
        self.action_space = spaces.Discrete(5)  # 0-3 move, 4 wait
        self.observation_space = spaces.Box(
            low=-self.grid_size,
            high=self.grid_size,
            shape=(5,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self._random_pos()
        self.pickup_pos = self._random_pos()
        self.delivery_pos = self._random_pos()

        self.obstacles = set()
        while len(self.obstacles) < self.n_obstacles:
            pos = tuple(self._random_pos())
            if pos not in [tuple(self.agent_pos), tuple(self.pickup_pos), tuple(self.delivery_pos)]:
                self.obstacles.add(pos)

        self.has_item = False
        self.steps = 0
        self.position_history = []
        self.recent_positions = []
        self.visited = set()
        self.visited.add(tuple(self.agent_pos))

        return self._get_obs(), {}

    def _random_pos(self):
        return np.random.randint(0, self.grid_size, size=2)

    def _get_obs(self):
        goal = self.pickup_pos if not self.has_item else self.delivery_pos
        rel_pos = goal - self.agent_pos
        return np.array([
            *self.agent_pos,
            *rel_pos,
            float(self.has_item)
        ], dtype=np.float32)
    
    def _detect_oscillation(self, new_pos):
        """Detect various oscillation patterns"""
        if len(self.recent_positions) < 3:
            return False
            
        # Check for immediate back-and-forth (A-B-A)
        if len(self.recent_positions) >= 2:
            if (new_pos == self.recent_positions[-2] and 
                self.recent_positions[-1] != new_pos):
                return True
        
        # Check for 3-cycle (A-B-C-A)
        if len(self.recent_positions) >= 3:
            if new_pos == self.recent_positions[-3]:
                return True
        
        # Check for 4-cycle patterns (A-B-C-D-A or A-B-A-C-A)
        if len(self.recent_positions) >= 4:
            if new_pos == self.recent_positions[-4]:
                return True
        
        # Check for repeated short sequences
        if len(self.recent_positions) >= 4:
            # Check if last 2 positions repeat: A-B-A-B
            if (self.recent_positions[-1] == self.recent_positions[-3] and
                self.recent_positions[-2] == self.recent_positions[-4]):
                return True
        
        return False
    
    def _get_penalty_for_revisit(self, pos):
        """Dynamic penalty based on how recently position was visited"""
        if pos not in self.position_history:
            return 0
        
        # Find most recent occurrence
        recent_index = len(self.position_history) - 1 - self.position_history[::-1].index(pos)
        steps_ago = len(self.position_history) - recent_index
        
        # Higher penalty for more recent revisits
        if steps_ago <= 2:
            return 8  # Very recent
        elif steps_ago <= 4:
            return 4  # Recent
        elif steps_ago <= 8:
            return 2  # Somewhat recent
        else:
            return 1  # Old visit
    
    def step(self, action):
        action = int(action)
        self.steps += 1
        reward = -1  # base step penalty
        terminated = False

        deltas = {
            0: (0, -1),  # Up
            1: (0, 1),   # Down
            2: (-1, 0),  # Left
            3: (1, 0),   # Right
            4: (0, 0),   # Wait
        }

        dx, dy = deltas.get(action, (0, 0))
        intended_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        new_x = max(0, min(self.grid_size - 1, intended_pos[0]))
        new_y = max(0, min(self.grid_size - 1, intended_pos[1]))
        new_pos = (new_x, new_y)

        # Handle obstacle collision
        if new_pos in self.obstacles:
            reward -= 5  # obstacle penalty
            fallback_found = False
            
            # Try alternative actions, but avoid oscillation
            for alt_action in [0, 1, 2, 3]:
                if alt_action == action:
                    continue
                dx_alt, dy_alt = deltas[alt_action]
                alt_x = self.agent_pos[0] + dx_alt
                alt_y = self.agent_pos[1] + dy_alt
                alt_pos = (alt_x, alt_y)
                
                if (0 <= alt_x < self.grid_size and
                    0 <= alt_y < self.grid_size and
                    alt_pos not in self.obstacles and
                    not self._detect_oscillation(alt_pos)):
                    new_pos = alt_pos
                    reward -= 1
                    fallback_found = True
                    break

            # Emergency diagonal escape if stuck
            if not fallback_found:
                diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dx_diag, dy_diag in diagonals:
                    diag_x = self.agent_pos[0] + dx_diag
                    diag_y = self.agent_pos[1] + dy_diag
                    diag_pos = (diag_x, diag_y)
                    if (0 <= diag_x < self.grid_size and
                        0 <= diag_y < self.grid_size and
                        diag_pos not in self.obstacles):
                        new_pos = diag_pos
                        reward -= 2
                        fallback_found = True
                        break

            if not fallback_found:
                new_pos = tuple(self.agent_pos)
                reward -= 2

        # Check for oscillation patterns
        if self._detect_oscillation(new_pos):
            reward -= 15  # Heavy penalty for oscillation
            
            # Force exploration by adding random noise to position
            # Try to find a non-oscillating position nearby
            alternatives = []
            for dx_rand in [-1, 0, 1]:
                for dy_rand in [-1, 0, 1]:
                    if dx_rand == 0 and dy_rand == 0:
                        continue
                    rand_x = self.agent_pos[0] + dx_rand
                    rand_y = self.agent_pos[1] + dy_rand
                    rand_pos = (rand_x, rand_y)
                    if (0 <= rand_x < self.grid_size and
                        0 <= rand_y < self.grid_size and
                        rand_pos not in self.obstacles and
                        not self._detect_oscillation(rand_pos)):
                        alternatives.append(rand_pos)
            
            if alternatives:
                new_pos = alternatives[np.random.randint(len(alternatives))]
                reward -= 1  # Additional penalty for forced random move

        # Update position tracking
        self.recent_positions.append(new_pos)
        if len(self.recent_positions) > self.recent_limit:
            self.recent_positions.pop(0)

        # Update agent position
        self.agent_pos = np.array(new_pos)
        pos_tuple = tuple(self.agent_pos)

        # Dynamic penalty for revisiting positions
        revisit_penalty = self._get_penalty_for_revisit(pos_tuple)
        reward -= revisit_penalty

        # Update position history
        self.position_history.append(pos_tuple)
        if len(self.position_history) > self.history_limit:
            self.position_history.pop(0)

        # Encourage exploration of new areas
        if pos_tuple not in self.visited:
            reward += 2  # Increased exploration reward
            self.visited.add(pos_tuple)

        # Reward shaping: movement toward goal
        current_target = self.pickup_pos if not self.has_item else self.delivery_pos
        dist = np.linalg.norm(self.agent_pos - current_target)
        
        # Calculate previous distance using the second-to-last position
        if len(self.recent_positions) >= 2:
            prev_pos = np.array(self.recent_positions[-2])
            prev_dist = np.linalg.norm(prev_pos - current_target)
            progress = prev_dist - dist
            reward += progress * 3  # Increased reward for progress
        
        # Pickup and delivery
        if not self.has_item and np.array_equal(self.agent_pos, self.pickup_pos):
            self.has_item = True
            reward += 8  # Increased pickup reward

        if self.has_item and np.array_equal(self.agent_pos, self.delivery_pos):
            reward += 15  # Increased delivery reward
            terminated = True

        if self.steps >= self.max_steps:
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            self._render_text()
        elif self.render_mode == "pygame" and PYGAME_AVAILABLE:
            self._render_pygame()
        elif self.render_mode == "pygame" and not PYGAME_AVAILABLE:
            print("[⚠️] PyGame not installed. Falling back to text rendering.")
            self._render_text()

    def _render_text(self):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        for ox, oy in self.obstacles:
            grid[oy, ox] = '#'
        grid[self.pickup_pos[1], self.pickup_pos[0]] = 'P'
        grid[self.delivery_pos[1], self.delivery_pos[0]] = 'D'
        grid[self.agent_pos[1], self.agent_pos[0]] = 'R'
        print("\n".join(" ".join(row) for row in grid))
        print(f"Has Item: {self.has_item} | Step: {self.steps}")
        print(f"Recent positions: {self.recent_positions[-5:]}")  # Debug info

    def _render_pygame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
            pygame.display.set_caption("Warehouse Robot")

        self.window.fill((240, 240, 240))
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

        # Draw recently visited positions with fading effect
        for i, pos in enumerate(self.recent_positions[-5:]):
            if pos != tuple(self.agent_pos):
                alpha = 50 + (i * 30)  # Fade effect
                color = (255, 200, 200, alpha)
                pygame.draw.rect(self.window, color[:3],
                                pygame.Rect(pos[0] * self.cell_size, pos[1] * self.cell_size, 
                                          self.cell_size, self.cell_size))

        for ox, oy in self.obstacles:
            pygame.draw.rect(self.window, (0, 0, 0),
                             pygame.Rect(ox * self.cell_size, oy * self.cell_size, self.cell_size, self.cell_size))

        px, py = self.pickup_pos
        dx, dy = self.delivery_pos
        ax, ay = self.agent_pos

        pygame.draw.rect(self.window, (0, 100, 255),
                         pygame.Rect(px * self.cell_size, py * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.window, (0, 200, 0),
                         pygame.Rect(dx * self.cell_size, dy * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.window, (255, 50, 50),
                         pygame.Rect(ax * self.cell_size, ay * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()
        pygame.event.pump()

    def close(self):
        if self.window:
            pygame.quit()