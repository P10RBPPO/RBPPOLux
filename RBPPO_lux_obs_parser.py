import numpy as np
import torch

def obs_parser(obs, custom_env):
    agent = "player_0"
    env_cfg = custom_env.env_cfg
    player_obs = obs[agent]
    
    # Ice map
    ice_map = player_obs["board"]["ice"]
    ice_tile_locations = np.argwhere(ice_map == 1)
    
    # Ore map
    ore_map = player_obs["board"]["ore"]
    ore_tile_locations = np.argwhere(ore_map == 1)
    
    # Observation space shape
    obs_vec = np.zeros(
        15,
    )
    
    # Factories
    factories = player_obs["factories"][agent]
    factory_vec = np.zeros(2)
    for factory_key in factories.keys():
        factory = factories.get(factory_key)
        factory_key = np.array(factory["pos"]) / env_cfg.map_size # Normalized first friendly factory position
        factory_cargo_vec = np.array(
            [
                factory["cargo"]["ice"] / 1000,
                factory["cargo"]["ore"] / 1000,
            ]
        )
        break
    
    # Units
    units = player_obs["units"][agent]
    for unit_key in units.keys():
        unit = units.get(unit_key)
        
        # cargo + power stored as values scaled from [0, 1]
        unit_cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
        unit_battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
        
        cargo_vec = np.array(
            [
                unit["power"] / unit_battery_cap,
                unit["cargo"]["ice"] / unit_cargo_space,
                unit["cargo"]["ore"] / unit_cargo_space,
            ]
        )
        
        unit_type = (0 if unit["unit_type"] == "LIGHT" else 1)
        
        # Normalize unit position
        pos = np.array(unit["pos"]) / env_cfg.map_size
        
        unit_vec = np.concatenate(
            [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
        )
        
        # Find closest ice tile to the unit
        ice_tile_locations = np.mean(
            (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
        )
        
        # Normalize ice tile location
        closest_ice_tile = (
            ice_tile_locations[np.argmin(ice_tile_locations)] / env_cfg.map_size
        )
        
        # Find closest ore tile to the unit
        ore_tile_locations = np.mean(
            (ore_tile_locations - np.array(unit["pos"])) ** 2, 1
        )
        
        # Normalize ore tile location
        closest_ore_tile = (
            ore_tile_locations[np.argmin(ore_tile_locations)] / env_cfg.map_size
        )
        
        # Combine observation vectors to a single np.array
        obs_vec = np.concatenate(
            [unit_vec, factory_vec - pos, factory_cargo_vec, closest_ice_tile - pos, closest_ore_tile - pos], axis=-1
        )
        break
    
    return obs_vec