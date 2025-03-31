import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Action space for both player 0 and player 1
def get_action_space(rule_based_early_step, map_size):
    action_space = spaces.Dict(
        {
            0: get_single_action_space(rule_based_early_step, map_size), 
            1: get_single_action_space(rule_based_early_step, map_size)
        }
    )
    return action_space

# Action space for only 1 player - might contain factory space, which should be removed
def get_single_action_space(rule_based_early_step, map_size):
    type_space = np.full((map_size, map_size), 7)
    direction_space = np.full((map_size, map_size), 5)
    resource_space = np.full((map_size, map_size), 5)
    amount_space = np.full((map_size, map_size), 10)
    repeat_space = np.full((map_size, map_size), 2)
    n_space = np.full((map_size, map_size), 20)
    unit_space = np.stack([type_space, direction_space, resource_space, amount_space, repeat_space, n_space])

    action_space = {
        "factory_act": spaces.MultiDiscrete(np.full((map_size, map_size), 4), dtype=np.float64), 
        "unit_act": spaces.MultiDiscrete(unit_space, dtype=np.float64), 
    }
    if not rule_based_early_step: # filter out probably, nicolai has logic for it already
        action_space.update(
            {
                "bid" : spaces.Discrete(11), 
                "factory_spawn": spaces.Dict(
                    {
                        "location": spaces.Discrete(map_size*map_size), 
                        "water": spaces.Box(low=0, high=1, shape=()), 
                        "metal": spaces.Box(low=0, high=1, shape=())
                    }
                )
            }
        )

    action_space = spaces.Dict(action_space)
    return action_space

# Complete game obs space
def get_observation_space(map_size):
    obs_space = spaces.Dict(
        {
            'player_0': get_single_observation_space(map_size), 
            'player_1': get_single_observation_space(map_size)
        }
    )
    return obs_space

# Game obs for only 1 player
def get_single_observation_space(map_size):
    global_feature_names = ['env_step',                 # 1000+10
                            'cycle',                    # 20
                            'hour',                     # 50
                            'daytime_or_night',         # 2
                            'num_factory_own',          # 5
                            'num_factory_enm',          # 5
                            'total_lichen_own',         # 48 * 48 * 100
                            'total_lichen_enm',         # 48 * 48 * 100
                            'factory_total_power_own',  # 9999
                            'factory_total_ice_own',    # 9999
                            'factory_total_water_own',  # 9999
                            'factory_total_ore_own',    # 9999
                            'factory_total_metal_own',  # 9999
                            'num_light_own',            # 9999
                            'num_heavy_own',            # 9999
                            'robot_total_power_own',    # 9999
                            'robot_total_ice_own',      # 9999
                            'robot_total_water_own',    # 9999
                            'robot_total_ore_own',      # 9999
                            'robot_total_metal_own',    # 9999
                            'factory_total_power_enm',  # 9999
                            'factory_total_ice_enm',    # 9999
                            'factory_total_water_enm',  # 9999
                            'factory_total_ore_enm',    # 9999
                            'factory_total_metal_enm',  # 9999
                            'num_light_enm',            # 9999
                            'num_heavy_enm',            # 9999
                            'robot_total_power_enm',    # 9999
                            'robot_total_ice_enm',      # 9999
                            'robot_total_water_enm',    # 9999
                            'robot_total_ore_enm',      # 9999
                            'robot_total_metal_enm']    # 9999
    global_feature_space = [
        1000, 
        20, 
        50, 
        2, 
        5, 
        5, 
        230400,
        230400, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999, 
        9999
    ]
    global_feature_space = spaces.MultiDiscrete(np.array(global_feature_space), dtype=np.float64)

    map_feature_names = {
        'ice': 9999, 
        'ore': 9999, 
        'rubble': 100, 
        'lichen': 100, 
        'lichen_strains': 11, # -1, 0, 1
        'lichen_strains_own': 2, 
        'lichen_strains_enm': 2, 
        'valid_region_indicator': 2, 
        'factory_id': 10, 
        'factory_power': 9999, 
        'factory_ice': 9999, 
        'factory_water': 9999, 
        'factory_ore': 9999, 
        'factory_metal': 9999, 
        'factory_own': 2, 
        'factory_enm': 2, 
        'factory_can_build_light': 2, 
        'factory_can_build_heavy': 2, 
        'factory_can_grow_lichen': 2, 
        'factory_water_cost': 9999, 
        'unit_id': 9999, 
        'unit_power': 9999, 
        'unit_ice': 9999, 
        'unit_water': 9999, 
        'unit_ore': 9999, 
        'unit_metal': 9999, 
        'unit_own': 2, 
        'unit_enm': 2, 
        'unit_light': 2, 
        'unit_heavy': 2
    }
    map_feature_space = np.tile(np.array(list(map_feature_names.values())).reshape(30, 1, 1), (1, map_size, map_size))
    map_feature_space = spaces.MultiDiscrete(map_feature_space, dtype=np.float64)

    action_feature_space = spaces.Dict(
        {
            'unit_indicator': spaces.MultiDiscrete(np.full((map_size, map_size), 2), dtype=np.float64),
            'type': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 7), dtype=np.float64),
            'direction': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 5), dtype=np.float64),
            'resource': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 5), dtype=np.float64),
            'amount': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 10), dtype=np.float64),
            'repeat': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 2), dtype=np.float64),
            'n': spaces.MultiDiscrete(np.full((map_size, map_size, 20), 9999), dtype=np.float64)
        }
    )
    obs_space = spaces.Dict(
        {
            'global_feature': global_feature_space, 
            'map_feature': map_feature_space, 
            'action_feature': action_feature_space
        }
    )
    return obs_space

class LuxEnv(gym.Env):
    
    def __init__(self):
        print("init")
        
    def step(self):
        print("step")
        
    def reset(self):
        print("reset")
