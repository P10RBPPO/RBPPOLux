from Controllers.FactoryController import FactoryController
from Controllers.RobotController import RobotController
from lux.kit import obs_to_game_state
import numpy as np


def parse_actions(custom_env, obs_dict, action_array, role):
    actions = dict()
    player = "player_0"
    env_cfg = custom_env.env_cfg
    
    # Get observations for a single player
    game_state = obs_to_game_state(custom_env.lux_env.state.env_steps, env_cfg, obs_dict[player])
    
    factory_controller = FactoryController(None, player)  # Initialize with None game state
    robot_controller = RobotController(None, player)  # Initialize with None game state
    robot_controller.update_game_state(game_state) # Update game_state
    
    # Get first robot unit from game_state
    units_dict = game_state.units[player]
    units = []
    for _, unit in units_dict.items():
        units.append(unit)
        break
    
    # Grab first unit from unit array if any units exist
    if len(units) == 0:
        unit = []
    else:
        unit = units[0]
    
    # Parse action array with information to get lux action and chosen action value for PPO
    robot_action, chosen_action, chosen_action_index = parse_action_array(action_array, robot_controller, unit, role)

    # remove this and replace with the single robot action above
    robot_actions = robot_controller.control_units(game_state=game_state)
    
    # Handle factory actions
    factory_actions = factory_controller.handle_factory_actions(player, env_cfg, game_state)
    
    combined_actions = {}
    combined_actions.update(robot_actions)
    combined_actions.update(factory_actions)
        
    actions[player] = combined_actions
        
    return actions, chosen_action, chosen_action_index


def parse_action_array(action_array, robot_controller, unit, role):
    chosen_action_index = np.argmax(action_array)
    chosen_action = action_array[chosen_action_index]
    
    # If we have a unit, assign a role to it and get an action
    if bool(unit):
        robot_controller.assign_role(unit.unit_id, role)
        role_type = robot_controller.unit_roles[unit.unit_id]
        
        robot_action = robot_action_parser(chosen_action_index, unit, robot_controller, role_type)
    
        return robot_action, chosen_action, chosen_action_index
    else:
        return np.array([]), np.array([0]), np.array([0])


def robot_action_parser(action_index, unit, robot_controller, role_type):
    robot_action = np.array([])

    if (action_index == 0):
        robot_action = robot_controller.move(unit, role_type) # Move
    elif (action_index == 1):
        robot_action = robot_controller.transfer(unit) # Transfer
    elif (action_index == 2):
        robot_action = robot_controller.pickup_power(unit) # Pickup
    elif (action_index == 3):
        robot_action = robot_controller.dig(unit) # Dig
    elif (action_index == 4):
        robot_action = robot_controller.recharge(unit) # Recharge
    elif (action_index == 5):
        robot_action = robot_controller.go_home(unit) # Go Home
    else:
        robot_action = [0, 0, 0, 0, 0, 1] # no-op 
    
    return robot_action

# Action array: [action_id, direction, resource, amount, repeat, n]
# Action direction: (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
# Action resource: (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 = power)