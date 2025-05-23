from Controllers.FactoryController import FactoryController
from Controllers.RobotController import RobotController
from lux.kit import obs_to_game_state
import numpy as np


# Creates a combined output of both factory and robot actions
def parse_all_actions(custom_env, obs_dict, action_index, role, robot_controller, factory_controller, factory_first_turn):
    actions = dict()
    player = "player_0"
    env_cfg = custom_env.env_cfg
    
    # Get observations for a single player
    game_state = obs_to_game_state(custom_env.lux_env.state.env_steps, env_cfg, obs_dict[player])
    
    robot_controller.update_game_state(game_state) # Update game_state
    
    # Get factory actions
    if (factory_first_turn == True):
        factory_actions = factory_controller.handle_factory_actions(player, env_cfg, game_state)
    else:
        factory_actions = dict()
    
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
    
    # Check if any units exist 
    if bool(unit):
        # Determine action based on index from categorical distribution
        robot_action, robot_action_q_length = robot_action_parser(action_index, robot_controller, unit, role, False)

        combined_actions = {}
        combined_actions.update({unit.unit_id: robot_action})
        combined_actions.update(factory_actions)
            
        actions[player] = combined_actions
            
        return actions, robot_action_q_length, False
    else:
        actions[player] = factory_actions
        return actions, 0, False

# Returns only an action set for the factory (for when macro actions are underway) - Alter code so no actions are provided after initial robot contruction
def factory_action_parser(custom_env, obs_dict, factory_controller, factory_first_turn):
    if (factory_first_turn == True):
        actions = dict()
        player = "player_0"
        env_cfg = custom_env.env_cfg
        
        # Get observations for a single player in game_state format
        game_state = obs_to_game_state(custom_env.lux_env.state.env_steps, env_cfg, obs_dict[player])
        
        factory_actions = factory_controller.handle_factory_actions(player, env_cfg, game_state)
        
        actions[player] = factory_actions
        return actions, False
    else:
        return {"player_0": {}}, False

def robot_action_parser(action_index, robot_controller, unit, role, live):
    robot_action = np.array([])
    
    if not live:
        if role != robot_controller.unit_roles[unit.unit_id]: 
            robot_controller.assign_role(unit.unit_id, role)
            role = robot_controller.unit_roles[unit.unit_id]
    
    if (action_index == 0):
        robot_action = robot_controller.go_home(unit) # Go Home
    elif (action_index == 1):
        robot_action = robot_controller.move(unit, "Ice Miner") # Move to Ice
    elif (action_index == 2):
        robot_action = robot_controller.move(unit, "Ore Miner") # Move to Ore
    elif (action_index == 3):
        robot_action = robot_controller.dig(unit) # Dig
    elif (action_index == 4):
        robot_action = robot_controller.pickup_power(unit) # Pickup
    elif (action_index == 5):
        robot_action = robot_controller.transfer(unit) # Transfer
    elif (action_index == 6):
        robot_action = robot_controller.recharge(unit) # Recharge
    else:
        robot_action = [0, 0, 0, 0, 0, 1] # no-op 
    
    return robot_action, len(robot_action)

# Action array: [action_id, direction, resource, amount, repeat, n]
# Action direction: (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
# Action resource: (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 = power)