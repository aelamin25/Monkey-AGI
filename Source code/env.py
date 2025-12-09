import numpy as np

class TapEnv:
    """
    Two-action episodic environment:
      0 = left (red tap)
      1 = right (blue tap)
    One tap gives reward +5, the other 0.
    """

    def __init__(self, sugar_tap=0, horizon=1):
        assert sugar_tap in (0, 1)
        self.sugar_tap = sugar_tap
        self.horizon = horizon
        self.t = 0

    def reset(self):
        self.t = 0
        return np.array([0.0], dtype=np.float32)

    def step(self, action):
        reward = 5.0 if action == self.sugar_tap else 0.0
        self.t += 1
        done = self.t >= self.horizon
        obs_next = np.array([0.0], dtype=np.float32)
        return obs_next, reward, done, {}

    def switch_sugar(self):
        """Swap reward to the opposite tap."""
        self.sugar_tap = 1 - self.sugar_tap
