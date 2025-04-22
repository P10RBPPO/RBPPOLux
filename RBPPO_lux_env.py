import gymnasium as gym
import numpy as np
from gymnasium import spaces
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import StatsStateDict
#from lux.kit import obs_to_game_state
from agent import Agent

class LuxCustomEnv(gym.Env):
    def __init__(self, env_config=None):
        super(LuxCustomEnv, self).__init__()
        
        # LuxAI_S2 env init
        self.env = LuxAI_S2(MIN_FACTORIES=1, MAX_FACTORIES=1)
        self.env.reset()
        
        # Create an Agent instance for bidding & factory placement
        self.agent = Agent("player_0", self.env.env_cfg)

        self.observation_space = spaces.Box(low=0, high=10, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        
        # keep for creating the true obs_space later
        #low = np.array([-10, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32)
        #high = np.array([10, 5, 5, 5, 1, 10, 10, 10, 10], dtype=np.float32)

        #self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        
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
    
    
    def step(self, action):
        
        # Fill enemy factories to survive 1000 turns
        opp_agent = self.agent.opp_player
        opp_factories = self.env.state.factories[opp_agent]
        
        for factory_key in opp_factories.keys():
            factory = opp_factories[factory_key]
            factory.cargo.water = 1000
        
        # Turn Tensors into an action before stepping
        
        obs, reward, done, truncated, info = self.env.step(action)
        print(obs)
        
        
        stats: StatsStateDict = self.env.state.stats[self.agent]
        # Rewards should be removed and customized to fit each role here before returning
        return obs, reward, done, truncated, info


    def render(self, mode='human'):
        return self.env.render(mode)


    def close(self):
        self.env.close()
