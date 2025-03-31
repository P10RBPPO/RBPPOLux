import heapq
import math
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys

class PathfindingResult:
    def __init__(self, path, move_cost, action_queue, tile_occupation):
        self.path = path
        self.total_move_cost = move_cost
        self.action_queue = action_queue
        self.tile_occupation = tile_occupation  # Dictionary of tile positions and the turns they are occupied

    # heuristic for A* search
    # we only care initially if we get closer to the target
    # move cost comes later
    @staticmethod
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # A* search algorithm
    # returns a PathfindingResult object
    @staticmethod
    def astar_search(unit, start, goal, game_state, current_turn):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        open_set = []
        heapq.heappush(open_set, (0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): PathfindingResult.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == tuple(goal):
                path = [current]  # Start with the goal
                total_move_cost = 0
                tile_occupation = {}  # Track when each tile is occupied
                turn = current_turn

                while current in came_from:
                    previous = came_from[current]
                    path.append(previous)
                    total_move_cost += PathfindingResult.move_cost(
                        game_state, np.array(previous), direction_to(np.array(previous), np.array(current)), unit
                    )
                    turn += 1
                    tile_occupation[tuple(previous)] = turn  # Mark the tile as occupied at this turn
                    current = previous

                path.reverse()  # Reverse the path to start from the initial position
                action_queue = PathfindingResult.build_action_queue(path, unit)
                return PathfindingResult(path, total_move_cost, action_queue, tile_occupation)

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # Skip invalid tiles
                if not PathfindingResult.is_valid_tile(game_state, neighbor, unit):
                    continue

                move_cost = PathfindingResult.move_cost(game_state, np.array(current), direction, unit)
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
        """
        Calculates the cost of moving from the current position in the given direction.
        Assumes the target tile is valid.
        """
        board = game_state.board
        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        target_pos = current_pos + move_deltas[direction]
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]
        return math.floor(unit.unit_cfg.MOVE_COST + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)
    
    @staticmethod
    def build_action_queue(path, unit):
        """
        Builds an action queue for the given path.
        Ensures the action queue does not exceed the maximum allowed length.
        """
        action_queue = []
        for step in range(len(path) - 1):  # Iterate through the path
            # Calculate direction from the current step to the next step
            direction = direction_to(np.array(path[step]), np.array(path[step + 1]))
            action_queue.append(unit.move(direction, repeat=0, n=1))
            
            # Stop adding actions if the max length is reached
            if len(action_queue) >= 20:
                break

        return action_queue

    @staticmethod
    def is_valid_tile(game_state, target_pos, unit):
        """
        Checks if a tile is valid for movement.
        Returns True if the tile is valid, False otherwise.
        """
        board = game_state.board
        if target_pos[0] < 0 or target_pos[1] < 0 or target_pos[1] >= len(board.rubble) or target_pos[0] >= len(board.rubble[0]):
            return False  # Out of bounds
        factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]
        if factory_there not in game_state.teams[unit.agent_id].factory_strains and factory_there != -1:
            return False  # Occupied by an enemy factory
        return True