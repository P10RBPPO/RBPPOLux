import gymnasium as gym
import numpy as np
from gymnasium import spaces
from luxai_s2.env import LuxAI_S2, EnvConfig
#from lux.config import EnvConfig
#from lux.kit import obs_to_game_state
from agent import Agent

class LuxCustomEnv(gym.Env):
    def __init__(self, env_config=None):
        super(LuxCustomEnv, self).__init__()
        
        # LuxAI_S2 env init
        self.env = LuxAI_S2()
        self.env.reset(seed=0)
        
        # Create an Agent instance for bidding & factory placement
        self.env_cfg = EnvConfig()
        self.agent = Agent("player_0", self.env_cfg)

        self.action_space = self.env.action_space(self.agent.player)
        self.observation_space = self.env.observation_space(self.agent.player)
        
    def reset(self, **kwargs):
        """Reset function handling bidding & factory placement."""
        obs, _ = self.env.reset(**kwargs)
        # Call 'early_setup' to handle bidding and factory placement
        while self.env.state.real_env_steps < 1:
            step = self.env.state.env_steps
            action = {agent: self.agent.early_setup(step, obs[agent]) for agent in self.env.agents}
            obs, _, _, _, _ = self.env.step(action)
        self.prev_obs = obs
        return obs, {}

    def reload_spaces(self):
        # keep for later
        #low = np.array([-10, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32)
        #high = np.array([10, 5, 5, 5, 1, 10, 10, 10, 10], dtype=np.float32)

        #self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=10, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()
