import gymnasium as gym
import numpy as np
from gymnasium import spaces
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import StatsStateDict, create_empty_stats
from agent import Agent

class LuxCustomEnv(gym.Env):
    def __init__(self, env_config=None):
        super(LuxCustomEnv, self).__init__()
        
        # LuxAI_S2 env init
        self.lux_env = LuxAI_S2(MIN_FACTORIES=1, MAX_FACTORIES=1)
        self.lux_env.reset()
        
        self.env_cfg = self.lux_env.env_cfg

        self.observation_space = spaces.Box(low=-999, high=999, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        
    def reset(self, **kwargs):
        """Reset function handling bidding & factory placement."""
        obs, _ = self.lux_env.reset(**kwargs)
        
        self.agents = {player: Agent(player, self.env_cfg) for player in self.lux_env.agents}
        
        # Call 'early_setup' to handle bidding and factory placement
        while self.lux_env.state.real_env_steps < 0:
            action = dict()
            for agent in self.lux_env.agents:
                step = self.lux_env.state.env_steps
                single_agent_action = self.agents[agent].early_setup(step, obs[agent])
                action[agent] = single_agent_action
            obs, _, _, _, _ = self.lux_env.step(action)
        self.prev_obs = obs
        return obs, {}
    
    def step(self, action):
        
        # Fill enemy factories to survive 1000 turns
        opp_agent = list(self.agents.keys())[1]
        opp_factories = self.lux_env.state.factories[opp_agent]
        
        for factory_key in opp_factories.keys():
            factory = opp_factories[factory_key]
            factory.cargo.water = 1000
        
        obs, reward, done, truncated, info = self.lux_env.step(action)
        
        # Collect metric stats for customized rewards
        player = list(self.agents.keys())[0]
        
        # Rewards should be removed and customized to fit each role here before returning
        return obs, reward, done, truncated, info


    def render(self, mode='human'):
        return self.lux_env.render(mode)


    def close(self):
        self.lux_env.close()
