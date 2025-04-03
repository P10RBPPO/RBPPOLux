import gymnasium as gym
from gymnasium import spaces
from luxai_s2.env import LuxAI_S2, EnvConfig
#from lux.config import EnvConfig
#from lux.kit import obs_to_game_state
from agent import Agent #Why tf does this not work!?

class LuxCustomEnv(gym.Env):
    def __init__(self, env_config=None):
        super(LuxCustomEnv, self).__init__()
        
        # LuxAI_S2 env init
        self.env = LuxAI_S2()
        
        # Create an Agent instance for bidding & factory placement
        self.env_cfg = EnvConfig()
        self.agent = Agent("player_0", self.env_cfg)

        self.action_space = self.env.action_space("player_0")
        self.observation_space = self.env.observation_space("player_0")
        
    def reset(self, **kwargs):
        """Reset function handling bidding & factory placement."""
        obs, _ = self.env.reset(**kwargs)

        # Call 'early_setup' to handle bidding and factory placement
        while self.env.state.real_env_steps < 0:
            step = self.env.state.env_steps
            action = {agent: self.agent.early_setup(step, obs[agent]) for agent in self.env.agents}
            obs, _, _, _, _ = self.env.step(action)

        self.prev_obs = obs
        return obs, {}

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()
