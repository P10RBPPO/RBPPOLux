import heapq
import math
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys

class PathfindingResult:
    def __init__(self, path, move_cost, action_queue):
        self.path = path
        self.move_cost = move_cost
        self.action_queue = action_queue

    # heuristic for A* search
    # we only care initially if we get closer to the target
    # move cost comes later
    @staticmethod
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # A* search algorithm
    # returns a PathfindingResult object
    @staticmethod
    def astar_search(unit, start, goal, game_state):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        map_width, map_height = 48, 48
        open_set = []
        heapq.heappush(open_set, (0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): PathfindingResult.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == tuple(goal):
                path = []
                total_move_cost = 0
                while current in came_from:
                    path.append(current)
                    total_move_cost += PathfindingResult.move_cost(game_state, np.array(came_from[current]), direction_to(np.array(came_from[current]), np.array(current)), unit)
                    current = came_from[current]
                path.reverse()
                action_queue = PathfindingResult.build_action_queue(path, unit)
                return PathfindingResult(path, total_move_cost, action_queue)

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # Check if neighbor is within bounds
                if not (0 <= neighbor[0] < map_width and 0 <= neighbor[1] < map_height):
                    continue

                move_cost = PathfindingResult.move_cost(game_state, np.array(current), direction, unit)
                if move_cost > 999999:
                    continue  # Skip this direction if move cost is None

                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + PathfindingResult.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print(f"no path found for {unit.unit_id}", file=sys.stderr)
        return None  # No path found

    # Calculate the cost of moving from current_pos to target_pos
    # This is taken from the library and modified to work with any location
    @staticmethod
    def move_cost(game_state, current_pos, direction, unit):
        board = game_state.board
        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        target_pos = current_pos + move_deltas[direction]
        if target_pos[0] < 0 or target_pos[1] < 0 or target_pos[1] >= len(board.rubble) or target_pos[0] >= len(board.rubble[0]):
            return 99999999
        factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]
        if factory_there not in game_state.teams[unit.agent_id].factory_strains and factory_there != -1:
            return 99999999
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]
        power_at_tile = 0
        return math.floor(unit.unit_cfg.MOVE_COST + power_at_tile + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)
    
    @staticmethod
    def build_action_queue(path, unit):
        action_queue = []
        for step in range(min(len(path) - 1, 20)):
            direction = direction_to(np.array(path[step]), np.array(path[step + 1]))
            action_queue.append(unit.move(direction, repeat=0, n=1))
        return action_queue