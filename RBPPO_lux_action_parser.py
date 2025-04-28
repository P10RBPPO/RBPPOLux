from Controllers.FactoryController import FactoryController
from Controllers.RobotController import RobotController
from lux.kit import obs_to_game_state


def parse_actions(custom_env, obs, action):
    actions = dict()
    player = "player_0"
    env_cfg = custom_env.env_cfg
    
    print(obs)
    
    # This shit expects obs not wrapped in a player element
    game_state = obs_to_game_state(custom_env.lux_env.state.env_steps, env_cfg, obs)
    
    print(obs)
    
    factory_controller = FactoryController(None, player)  # Initialize with None game state
    robot_controller = RobotController(None, player)  # Initialize with None game state

    robot_actions = robot_controller.control_units(game_state=game_state)
    
    factory_actions = factory_controller.handle_factory_actions(player, env_cfg, game_state)
    
    combined_actions = {}
    combined_actions.update(robot_actions)
    combined_actions.update(factory_actions)
        
    actions[player] = combined_actions
        
    return actions