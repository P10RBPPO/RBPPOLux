from Controllers.FactoryController import FactoryController
from Controllers.RobotController import RobotController
from lux.kit import obs_to_game_state
import numpy as np


def parse_actions(custom_env, obs_dict, action_array):
    actions = dict()
    player = "player_0"
    env_cfg = custom_env.env_cfg
    
    # This shit expects obs not wrapped in a player element
    game_state = obs_to_game_state(custom_env.lux_env.state.env_steps, env_cfg, obs_dict[player])
    
    factory_controller = FactoryController(None, player)  # Initialize with None game state
    robot_controller = RobotController(None, player)  # Initialize with None game state

    # Get the first robot type from obs_dict
    robot_type = 0
    player_obs = obs_dict[player]
    units = player_obs["units"][player]
    for unit_key in units.keys():
        unit = units.get(unit_key)
        robot_type = (0 if unit["unit_type"] == "LIGHT" else 1)
        break
    
    # Get friendly factories (Expected: 1)
    _, factory = robot_controller.get_factories(game_state)
    
    # Parse action array with information to get lux action and chosen action value for PPO
    robot_action, chosen_action = parse_action_array(action_array, factory, robot_controller, robot_type)

    # remove this and replace with the single robot action above
    robot_actions = robot_controller.control_units(game_state=game_state)
    
    # Handle factory actions
    factory_actions = factory_controller.handle_factory_actions(player, env_cfg, game_state)
    
    combined_actions = {}
    combined_actions.update(robot_actions)
    combined_actions.update(factory_actions)
        
    actions[player] = combined_actions
        
    return actions, chosen_action


def parse_action_array(action_array, factory, robot_controller, robot_type):
    abs_action_array = np.abs(action_array)
    chosen_action_index = np.argmin(abs_action_array)
    chosen_action = action_array[chosen_action_index]
    
    role_type = 0 # fix later, should be determined by robot_controller
    role_type_cargo = (1 if role_type == "Ore Miner" else 0)
    
    robot_action = robot_action_parser(chosen_action_index, factory, role_type, role_type_cargo, robot_type)
    
    return robot_action, chosen_action


def robot_action_parser(action_index, factory, role_type, role_type_cargo, robot_type):
    robot_action = np.array([])

    if (action_index == 0):
        robot_action = [0, 0, 0, 0, 0, 1] # Move (fix after meeting)
    elif (action_index == 1):
        robot_action = [1, 0, role_type, role_type_cargo, 0, 1] # Transfer
    elif (action_index == 2):
        robot_action = np.array([2, 0, 4, (factory[0].power * 0.2), 0, 1]) # Pickup
    elif (action_index == 3):
        robot_action = np.array([3, 0, 0, 0, 0, 1]) # Dig
    elif (action_index == 4):
        robot_action = [5, 0, 0, robot_type, 0, 1] # Recharge
    elif (action_index == 5):
        robot_action = [0, 0, 0, 0, 0, 1] # Go Home (fix after meeting)
    else:
        robot_action = [0, 0, 0, 0, 0, 1] # no-op 
    
    return robot_action

# Action array: [action_id, direction, resource, amount, repeat, n]
# Action direction: (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
# Action resource: (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 = power)