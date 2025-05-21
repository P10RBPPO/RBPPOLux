import sys
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
        13,
    )
    
    # Factories
    factories = player_obs["factories"][agent]
    factory_pos_vec = np.zeros(2)
    factory_cargo_vec = np.zeros(3)
    for factory_key in factories.keys():
        factory = factories.get(factory_key)
        factory_pos_vec = np.array(factory["pos"]) / env_cfg.map_size # Normalized first friendly factory position
        factory_cargo_vec = np.array(
            [
                factory["power"] / 1000,
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
        
        unit_cargo_vec = np.array(
            [
                unit["power"] / unit_battery_cap,
                unit["cargo"]["ice"] / unit_cargo_space,
                unit["cargo"]["ore"] / unit_cargo_space,
            ]
        )
        
        unit_type = (0 if unit["unit_type"] == "LIGHT" else 1)
        
        # Raw and normalized unit position
        unit_pos_raw = np.array(unit["pos"])
        unit_pos = unit_pos_raw / env_cfg.map_size
        
        # Squared euclidean distance to factory closest to the unit (assuming 1 factory)
        factory_distance = np.sum((factory_pos_vec - unit_pos) ** 2)
        
        # Append unit information together
        unit_vec = np.concatenate(
            [unit_pos, [unit_type], unit_cargo_vec, [unit["team_id"]]], axis=-1
        )
        
        # Find the closest ice tile to the unit and return the squared euclidean distance to it
        closest_ice_tile_distance = (
            np.min(np.sum((ice_tile_locations - unit_pos_raw) ** 2, axis=1)) / (env_cfg.map_size ** 2)
        )
        
        # Find the closest ore tile to the unit and return the squared euclidean distance to it
        closest_ore_tile_distance = (
            np.min(np.sum((ore_tile_locations - unit_pos_raw) ** 2, axis=1)) / (env_cfg.map_size ** 2)
        )
        
        # Combine observation vectors to a single np.array
        obs_vec = np.concatenate(
            [unit_vec, [factory_distance], factory_cargo_vec, [closest_ice_tile_distance], [closest_ore_tile_distance]], axis=-1
        )
        break
    
    return obs_vec


def live_obs_parser(player_obs, robot_unit, factory_unit):
    env_cfg = robot_unit.env_cfg
    
    # Ice map
    ice_map = player_obs["board"]["ice"]
    ice_tile_locations = np.argwhere(ice_map == 1)
    
    # Ore map
    ore_map = player_obs["board"]["ore"]
    ore_tile_locations = np.argwhere(ore_map == 1)
    
    # Observation space shape
    obs_vec = np.zeros(
        13,
    )
    
    # Factories
    factory_pos_vec = np.zeros(2)
    factory_cargo_vec = np.zeros(3)
    
    factory_pos_vec = np.array(factory_unit.pos) / env_cfg.map_size # Normalized nearest friendly factory position
    factory_cargo_vec = np.array(
        [
            factory_unit.power / 1000,
            factory_unit.cargo.ice / 1000,
            factory_unit.cargo.ore / 1000,
        ]
    )
    
    # Units
    # cargo + power stored as values scaled from [0, 1]
    unit_cargo_space = env_cfg.ROBOTS[robot_unit.unit_type].CARGO_SPACE
    unit_battery_cap = env_cfg.ROBOTS[robot_unit.unit_type].BATTERY_CAPACITY
    
    unit_cargo_vec = np.array(
        [
            robot_unit.power / unit_battery_cap,
            robot_unit.cargo.ice / unit_cargo_space,
            robot_unit.cargo.ore / unit_cargo_space,
        ]
    )
    
    unit_type = (0 if robot_unit.unit_type == "LIGHT" else 1)
    
    # Raw and normalized unit position
    unit_pos_raw = np.array(robot_unit.pos)
    unit_pos = unit_pos_raw / env_cfg.map_size
    
    # Squared euclidean distance to factory closest to the unit (assuming 1 factory)
    factory_distance = np.sum((factory_pos_vec - unit_pos) ** 2)
    
    # Append unit information together
    unit_vec = np.concatenate(
        [unit_pos, [unit_type], unit_cargo_vec, [robot_unit.team_id]], axis=-1
    )
    
    # Find the closest ice tile to the unit and return the squared euclidean distance to it
    closest_ice_tile_distance = (
        np.min(np.sum((ice_tile_locations - unit_pos_raw) ** 2, axis=1)) / (env_cfg.map_size ** 2)
    )
    
    # Find the closest ore tile to the unit and return the squared euclidean distance to it
    closest_ore_tile_distance = (
        np.min(np.sum((ore_tile_locations - unit_pos_raw) ** 2, axis=1)) / (env_cfg.map_size ** 2)
    )
    
    # Combine observation vectors to a single np.array
    obs_vec = np.concatenate(
        [unit_vec, [factory_distance], factory_cargo_vec, [closest_ice_tile_distance], [closest_ore_tile_distance]], axis=-1
    )
    
    return obs_vec