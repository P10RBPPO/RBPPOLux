import heapq
import math
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys
class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

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
        
        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """

        game_state = obs_to_game_state(step, self.env_cfg, obs)
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
        for unit_id, unit in units.items():

            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

                # previous ice mining code
                if unit.cargo.ice < 40:
                    ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                    closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                    if np.all(closest_ice_tile == unit.pos):
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                # else if we have enough ice, we go back to the factory and dump it.
                elif unit.cargo.ice >= 40:
                    direction = direction_to(unit.pos, closest_factory_tile)
                    if adjacent_to_factory:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
        return actions

    # A* search algorithm
    # returns a path from start to goal
    def astar_search(self, unit, start, goal, game_state):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        map_width, map_height = self.env_cfg.map_size, self.env_cfg.map_size
        open_set = []
        heapq.heappush(open_set, (0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == tuple(goal):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # Check if neighbor is within bounds
                if not (0 <= neighbor[0] < map_width and 0 <= neighbor[1] < map_height):
                    #print(f"neighbor {neighbor} is out of bounds", file=sys.stderr)
                    continue

                move_cost = self.move_cost(game_state, current, direction, unit)
                if move_cost > 999999:
                    # Uncomment the line below for debugging
                    # print(f"move cost {move_cost} for {unit.unit_id} at {start} searching {current} with neighbor {neighbor}", file=sys.stderr)
                    continue  # Skip this direction if move cost is None

                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print(f"no path found for {unit.unit_id}", file=sys.stderr)
        return None  # No path found

    # Calculate the cost of moving from current_pos to target_pos
    # This is taken from the library and modified to work with any location
    def move_cost(self, game_state, current_pos, direction, unit):
        board = game_state.board
        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        target_pos = current_pos + move_deltas[direction]
        if target_pos[0] < 0 or target_pos[1] < 0 or target_pos[1] >= len(board.rubble) or target_pos[0] >= len(board.rubble[0]):
            # print("Warning, tried to get move cost for going off the map", file=sys.stderr)
            return 99999999
        factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]
        if factory_there not in game_state.teams[unit.agent_id].factory_strains and factory_there != -1:
            # print("Warning, tried to get move cost for going onto a opposition factory", file=sys.stderr)
            return 99999999
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]
        power_at_tile = 0 # self.get_unit_power_on_tile(target_pos, game_state.units[unit.agent_id], unit.team_id)
        return math.floor(unit.unit_cfg.MOVE_COST + power_at_tile + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)