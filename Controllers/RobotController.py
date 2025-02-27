import heapq
import math
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys
import PathfindingResult

class RobotController:
    def __init__(self, game_state, player):
        self.game_state = game_state
        self.player = player
        self.units = {}
        self.unit_types = {}
        self.unit_roles = {}

    def add_unit(self, unit_id, unit, unit_type):
        self.units[unit_id] = unit
        self.unit_types[unit_id] = unit_type

    def assign_role(self, unit_id, role):
        self.unit_roles[unit_id] = role
    
    def update_game_state(self, new_game_state):
        self.game_state = new_game_state
    

    def control_units(self, actions):
        factory_tiles, factory_units = self.get_factories(self.game_state)
        ice_tile_locations = self.get_ice_tile_locations(self.game_state)

        for unit_id, unit in self.units.items():
            role = self.unit_roles.get(unit_id, None)
            if role == "Ice Miner":
                self.control_ice_miner(unit_id, unit, factory_tiles, factory_units, ice_tile_locations, actions)
        
        # Resolve conflicts before returning actions
        actions = self.resolve_conflicts(actions)
        return actions

    def control_ice_miner(self, unit_id, unit, factory_tiles, factory_units, ice_tile_locations, actions):
        closest_factory = None
        adjacent_to_factory = False
        if len(factory_tiles) > 0:
            factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
            closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
            closest_factory = factory_units[np.argmin(factory_distances)]
            adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

            if unit.cargo.ice < 40:
                ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                if np.all(closest_ice_tile == unit.pos):
                    if unit.power >= unit.dig_cost(self.game_state) + unit.action_queue_cost(self.game_state):
                        actions[unit_id] = [unit.dig(repeat=0, n=1)]
                else:
                    direction = direction_to(unit.pos, closest_ice_tile)
                    move_cost = unit.move_cost(self.game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(self.game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
            elif unit.cargo.ice >= 40:
                direction = direction_to(unit.pos, closest_factory_tile)
                if adjacent_to_factory:
                    if unit.power >= unit.action_queue_cost(self.game_state):
                        actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
                else:
                    move_cost = unit.move_cost(self.game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(self.game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
    
    def resolve_conflicts(self, actions):
        target_positions = {}
        resolved_actions = {}

        for unit_id, action_list in actions.items():
            if action_list:
                action = action_list[0]
                if isinstance(action, np.ndarray) and action[0] == 0:  # Check if it's a move action
                    direction = action[1]
                    target_pos = (self.units[unit_id].pos[0] + direction[0], self.units[unit_id].pos[1] + direction[1])
                    if target_pos not in target_positions:
                        target_positions[target_pos] = []
                    target_positions[target_pos].append(unit_id)

        for target_pos, unit_ids in target_positions.items():
            if len(unit_ids) > 1:
                unit_ids.sort(key=lambda uid: (self.unit_types[uid] != 'HEAVY', int(uid.split('_')[1])))
                resolved_actions[unit_ids[0]] = actions[unit_ids[0]]
                for uid in unit_ids[1:]:
                    resolved_actions[uid] = []  # Cancel the move action by doing nothing
            else:
                resolved_actions[unit_ids[0]] = actions[unit_ids[0]]

        for unit_id in actions:
            if unit_id not in resolved_actions:
                resolved_actions[unit_id] = actions[unit_id]

        return resolved_actions

    def get_factories(self, game_state):
        factories = game_state.factories[self.player]
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)
        return factory_tiles, factory_units
    
    def get_ice_tile_locations(self, game_state):
        ice_map = game_state.board.ice
        return np.argwhere(ice_map == 1)

    def get_rubble_tile_locations(self, game_state):
        rubble_map = game_state.board.rubble
        return np.argwhere(rubble_map == 1)

    def get_ore_tile_locations(self, game_state):
        ore_map = game_state.board.ore
        return np.argwhere(ore_map == 1)