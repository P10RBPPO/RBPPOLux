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
        water_left = game_state.teams[player].water
        metal_left = game_state.teams[player].metal
        factories_to_place = game_state.teams[player].factories_to_place
        my_turn_to_place = my_turn_to_place_factory(game_state.teams[player].place_first, step)

        if factories_to_place > 0 and my_turn_to_place:
            potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
            spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
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