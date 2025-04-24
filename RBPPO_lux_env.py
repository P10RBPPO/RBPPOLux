import gymnasium as gym
import numpy as np
from gymnasium import spaces
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import StatsStateDict
from agent import Agent

class LuxCustomEnv(gym.Env):
    def __init__(self, env_config=None):
        super(LuxCustomEnv, self).__init__()
        
        # LuxAI_S2 env init
        self.lux_env = LuxAI_S2(MIN_FACTORIES=1, MAX_FACTORIES=1)
        self.lux_env.reset()
        
        self.env_cfg = self.lux_env.env_cfg
        self.agent = {}

        self.observation_space = spaces.Box(low=0, high=10, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        
        # keep for creating the true obs_space later
        #low = np.array([-10, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32)
        #high = np.array([10, 5, 5, 5, 1, 10, 10, 10, 10], dtype=np.float32)

        #self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        
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
                #action = {agent: self.agent.early_setup(step, obs[agent]) for agent in self.lux_env.agents}
            obs, _, _, _, _ = self.lux_env.step(action)
        self.prev_obs = obs
        return obs[list(self.agents.keys())[0]], {}        
    
    
    def step(self, action):
        
        # Fill enemy factories to survive 1000 turns
        opp_agent = list(self.agents.keys())[1]
        opp_factories = self.lux_env.state.factories[opp_agent]
        
        for factory_key in opp_factories.keys():
            factory = opp_factories[factory_key]
            factory.cargo.water = 1000
        
        # Turn Tensors into an action before stepping (ONLY RELEVANT FOR ROBOTS!)
        
        obs, reward, done, truncated, info = self.lux_env.step(action)
        
        print(list(self.lux_env.state.stats.keys()))
        
        player = list(self.agents.keys())[0]
        if player in self.lux_env.state.stats:
            stats: StatsStateDict = self.lux_env.state.stats[player]
            print(stats)
            
        # Rewards should be removed and customized to fit each role here before returning
        return obs, reward, done, truncated, info


    def render(self, mode='human'):
        return self.lux_env.render(mode)


    def close(self):
        self.lux_env.close()
