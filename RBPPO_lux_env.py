import copy
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from luxai_s2.env import LuxAI_S2
import torch
from RBPPO_lux_obs_parser import obs_parser
from RBPPO_lux_reward_parser import reward_parser
from agent import Agent

class LuxCustomEnv(gym.Env):
    def __init__(self, env_config=None):
        super(LuxCustomEnv, self).__init__()
        
        # LuxAI_S2 env init
        self.lux_env = LuxAI_S2(MIN_FACTORIES=1, MAX_FACTORIES=1)
        self.lux_env.reset()
        
        self.env_cfg = self.lux_env.env_cfg

        self.observation_space = spaces.Box(low=-999, high=999, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Discrete(7)
        
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
    
    def step(self, action, prev_obs, custom_env, role, shaping_level):
        
        # Fill enemy factories to survive 1000 turns
        opp_agent = list(self.agents.keys())[1]
        opp_factories = self.lux_env.state.factories[opp_agent]
        
        for factory_key in opp_factories.keys():
            factory = opp_factories[factory_key]
            factory.cargo.water = 1000
        
        # Fill friendly factory to survive 1000 turns if role is Ore miner (so it can survive 1000 turns to train in)
        if role == "Ore Miner":
            agent = list(self.agents.keys())[0]
            factories = self.lux_env.state.factories[agent]
            
            for factory_key in factories.keys():
                factory = factories[factory_key]
                factory.cargo.water = 1000
        
        # Possibly save reward here, as it returns reward for surviving 1000 turns or dying
        obs, _, done, truncated, info = self.lux_env.step(action)
        
        new_obs = torch.tensor(obs_parser(copy.deepcopy(obs), custom_env), dtype=torch.float)
        
        reward, reward_0, reward_1, reward_2 = reward_parser(prev_obs, new_obs, role, shaping_level)

        return obs, reward, done, truncated, info, reward_0, reward_1, reward_2


    def render(self, mode='human'):
        return self.lux_env.render(mode)


    def close(self):
        self.lux_env.close()
