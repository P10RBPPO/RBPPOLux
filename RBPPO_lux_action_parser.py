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

    robot_action, chosen_action = parse_action_array(action_array)

    robot_actions = robot_controller.control_units(game_state=game_state)
    
    factory_actions = factory_controller.handle_factory_actions(player, env_cfg, game_state)
    
    combined_actions = {}
    combined_actions.update(robot_actions)
    combined_actions.update(factory_actions)
        
    actions[player] = combined_actions
        
    return actions, chosen_action

def parse_action_array(action_array):
    chosen_action_index = np.argmin(action_array)
    chosen_action = action_array[chosen_action_index]
    
    robot_action = robot_action_parser(chosen_action_index)
    
    return robot_action, chosen_action

def robot_action_parser(action_index):
    robot_action = []
    
    if (action_index == 0):
        robot_action = [0] # Move
    elif (action_index == 1):
        robot_action = [1] # Dig
    elif (action_index == 2):
        robot_action = [2] # Pickup
    elif (action_index == 3):
        robot_action = [3] # Transfer
    elif (action_index == 4):
        robot_action = [4] # Recharge
    elif (action_index == 5):
        robot_action = [5] # Go Home
    else:
        robot_action = [] # no-op i guess?
    
    return robot_action

# def _is_move_action(self, id):
#         return id < self.move_dim_high

# def _get_move_action(self, id):
#     # move direction is id + 1 since we don't allow move center here
#     return np.array([0, id + 1, 0, 0, 0, 1])

# def _is_transfer_action(self, id):
#     return id < self.transfer_dim_high

# def _get_transfer_action(self, id):
#     id = id - self.move_dim_high
#     transfer_dir = id % 5
#     return np.array([1, transfer_dir, 0, self.env_cfg.max_transfer_amount, 0, 1])

# def _is_pickup_action(self, id):
#     return id < self.pickup_dim_high

# def _get_pickup_action(self, id):
#     return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

# def _is_dig_action(self, id):
#     return id < self.dig_dim_high

# def _get_dig_action(self, id):
#     return np.array([3, 0, 0, 0, 0, 1])