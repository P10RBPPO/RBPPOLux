import heapq
import math
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys
import PathfindingResult
import Controllers.RobotController as RobotController

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.robot_controller = RobotController.RobotController(None)  # Initialize with None game state

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
                spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        self.robot_controller.update_game_state(game_state)  # Update the RobotController with the new game_state

        factories = game_state.factories[self.player]
        game_state.teams[self.player].place_first
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            if factory.water_cost(game_state) <= factory.cargo.water / 5 - 200:
                actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)

        # Add units to the RobotController
        for unit_id, unit in units.items():
            self.robot_controller.add_unit(unit_id, unit, unit.unit_type)
            self.robot_controller.assign_role(unit_id, "Ice Miner")  # Assign role as Ice Miner for now

        # Control units using the RobotController
        actions = self.robot_controller.control_units(actions)

        return actions

    def merge_action_queues(self, first, second):
        return first + second

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