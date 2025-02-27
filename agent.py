import heapq
import math
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys
import PathfindingResult
import Controllers.RobotController as RobotController
import Controllers.FactoryController as FactoryController

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.robot_controller = RobotController.RobotController(None, player)  # Initialize with None game state
        self.factory_controller = FactoryController.FactoryController(None)  # Initialize with None game state

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            return self.factory_controller.place_factory(self.player, step, self.env_cfg, obs)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        self.robot_controller.update_game_state(game_state)  # Update the RobotController with the new game_state
        self.factory_controller.update_game_state(game_state) # Update the FactoryController with the new game_state

        # Control units using the RobotController
        robot_actions = self.robot_controller.control_units(actions)

        # Handle factory actions using the FactoryController
        factory_actions = self.factory_controller.handle_factory_actions(self.player, self.env_cfg, game_state, actions)
        
        # Merge the two action queues
        actions = self.merge_action_queues(factory_actions, robot_actions)

        return actions

    def merge_action_queues(self, first, second):
        first.update(second)
        return first

    def goto_closest_ice(self, unit, game_state, numdigs):
        ice_tile_locations = np.argwhere(game_state.board.ice == 1)
        ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
        closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
        astar = PathfindingResult.astar_search(unit, unit.pos, closest_ice_tile, game_state)
        action_queue = astar.action_queue
        return action_queue

    def return_to_factory(self, unit, factory, game_state):
        astar = PathfindingResult.astar_search(unit, unit.pos, factory.pos, game_state)
        action_queue = astar.action_queue
        action_queue.append(unit.transfer(direction_to(unit.pos, factory.pos), 0, unit.cargo.ice, repeat=0))
        return action_queue