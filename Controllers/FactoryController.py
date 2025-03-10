import heapq
import math
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys
import PathfindingResult

class FactoryController:
    def __init__(self, game_state, player):
        self.game_state = game_state
        self.player = player
        self.factories = {}
        self.factory_tiles = []
        self.factory_units = []

    def update_game_state(self, new_game_state):
        self.game_state = new_game_state
        # Remove dead factories
        current_factory_ids = set(new_game_state.factories[self.player].keys())
        self.factories = {factory_id: factory for factory_id, factory in self.factories.items() if factory_id in current_factory_ids}
        # Update existing factories and add new factories
        for factory_id, factory in new_game_state.factories[self.player].items():
            self.factories[factory_id] = factory

    def place_factory(self, player, step, env_cfg, obs):
        game_state = obs_to_game_state(step, env_cfg, obs)
        #water_left = game_state.teams[player].water
        #metal_left = game_state.teams[player].metal
        factories_to_place = game_state.teams[player].factories_to_place
        my_turn_to_place = my_turn_to_place_factory(game_state.teams[player].place_first, step)

        if factories_to_place > 0 and my_turn_to_place:
            #potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
            #spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
            spawn_loc = self.find_best_factory_location(obs)
            return dict(spawn=spawn_loc, metal=150, water=150)
        return dict()
    
    def handle_factory_actions(self, player, env_cfg, game_state, actions):
        factories = game_state.factories[player]
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if factory.power >= env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            if factory.water_cost(game_state) <= factory.cargo.water / 5 - 200:
                actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)
        self.factory_tiles = factory_tiles
        self.factory_units = factory_units
        return actions
    
    # Find the best location to place a factory
    # Definition of best in this case means adjacent to ice and far from enemy factories
    def find_best_factory_location(self, obs):
        potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
        ice_tiles = np.array(list(zip(*np.where(obs["board"]["ice"] == 1))))
        ore_tiles = np.array(list(zip(*np.where(obs["board"]["ore"] == 1))))
        enemy_factories = np.array([factory.pos for factory in self.game_state.factories[self.opp_player].values()])

        best_location = None
        best_cost = float('inf')

        for spawn in potential_spawns:
            # Calculate distance to nearest ice tile
            ice_distances = np.linalg.norm(ice_tiles - spawn, axis=1)
            min_ice_distance = np.min(ice_distances)

            # Calculate distance to nearest ore tile
            ore_distances = np.linalg.norm(ore_tiles - spawn, axis=1)
            min_ore_distance = np.min(ore_distances)

            # Calculate distance to nearest enemy factory
            if len(enemy_factories) > 0:
                enemy_distances = np.linalg.norm(enemy_factories - spawn, axis=1)
                min_enemy_distance = np.min(enemy_distances)
            else:
                min_enemy_distance = float('inf')

            # Combine costs with adjusted weights
            total_cost = (min_ice_distance * 0.6) + (min_ore_distance * 0.4) + (1 / (min_enemy_distance + 1) * 0.1)  # Adjust weights as needed

            if total_cost < best_cost:
                best_cost = total_cost
                best_location = spawn

        return best_location