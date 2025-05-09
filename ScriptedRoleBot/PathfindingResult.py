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
        self.total_move_cost = move_cost
        self.action_queue = action_queue

    # heuristic for A* search
    # we only care initially if we get closer to the target
    # move cost comes later
    @staticmethod
    def heuristic(a, b, unit_type):
        if unit_type == "HEAVY":
            return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * 20
        elif unit_type == "LIGHT":
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # A* search algorithm
    # returns a PathfindingResult object
    @staticmethod
    def astar_search(unit, start, goal, game_state):
        """
        A* search algorithm to find the shortest path from start to goal.
        Returns a PathfindingResult object or None if no path is found.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Check if the goal state is valid
        if not PathfindingResult.is_valid_tile(game_state, goal, unit):
            #print(f"Goal {goal} is invalid for unit {unit.unit_id}. Returning empty path.", file=sys.stderr)
            return None  # Return immediately if the goal is invalid

        open_set = []
        heapq.heappush(open_set, (0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): PathfindingResult.heuristic(start, goal, unit.unit_type)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == tuple(goal):
                path = [current]  # Start with the goal
                total_move_cost = 0

                while current in came_from:
                    previous = came_from[current]
                    path.append(previous)
                    current = previous

                path.reverse()  # Reverse the path to start from the initial position

                # Calculate the total move cost after reversing the path
                for i in range(len(path) - 1):
                    # Calculate the direction from the current step to the next step
                    direction = PathfindingResult.my_direction_to(np.array(path[i]), np.array(path[i + 1]))
                    # Add the move cost for this step
                    step_cost = PathfindingResult.move_cost(game_state, np.array(path[i]), direction, unit, include_turn_cost=False)
                    total_move_cost += step_cost

                action_queue = PathfindingResult.build_action_queue(path, unit)
                return PathfindingResult(path, total_move_cost, action_queue)

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # Skip invalid tiles
                if not PathfindingResult.is_valid_tile(game_state, neighbor, unit):
                    continue

                move_cost = PathfindingResult.move_cost(game_state, np.array(current), direction, unit, include_turn_cost=True)
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + PathfindingResult.heuristic(neighbor, goal, unit.unit_type)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print(f"No path found for unit {unit.unit_id} to goal {goal}.", file=sys.stderr)
        return None  # No path found

    # Calculate the cost of moving from current_pos to target_pos
    # This is taken from the library and modified to work with any location
    @staticmethod
    def move_cost(game_state, current_pos, direction, unit, include_turn_cost=False):
        """
        Calculates the cost of moving from the current position in the given direction.
        If `include_turn_cost` is True, adds a small "turn cost" to prioritize faster paths.
        """

        board = game_state.board
        #move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        target_pos = current_pos + direction
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]

        # Base move cost
        move_cost = unit.unit_cfg.MOVE_COST + unit.action_queue_cost(game_state) + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target

        # Add a small turn cost if specified
        if include_turn_cost:
            move_cost += 0  # Example turn cost, adjust as needed
        
        return math.floor(move_cost)
    
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

        # Check if the tile is out of bounds
        if target_pos[0] < 0 or target_pos[1] < 0 or target_pos[1] >= len(board.rubble) or target_pos[0] >= len(board.rubble[0]):
            return False  # Out of bounds

        # Check if the tile is occupied by an enemy factory
        factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]
        if factory_there not in game_state.teams[unit.agent_id].factory_strains and factory_there != -1:
            return False  # Occupied by an enemy factory

        # Check if any friendly unit (excluding the original unit) is within 3 tiles of the target position
        for unit_id, other_unit in game_state.units[unit.agent_id].items():
            if unit_id == unit.unit_id:
                continue  # Skip the original unit
            distance = abs(other_unit.pos[0] - target_pos[0]) + abs(other_unit.pos[1] - target_pos[1])
            if distance <= 3 and np.array_equal(other_unit.pos, target_pos):
                return False  # Tile is invalid if occupied by another unit within 3 tiles

        return True
    
    @staticmethod
    def my_direction_to(src, target):
        """
        Calculates the direction from src to target as a tuple.
        Returns one of the directions: (0, 1), (1, 0), (0, -1), (-1, 0).
        """
        ds = target - src
        dx = ds[0]
        dy = ds[1]

        if dx == 0 and dy < 0:
            return (0, -1)  # Up
        elif dx > 0 and dy == 0:
            return (1, 0)  # Right
        elif dx == 0 and dy > 0:
            return (0, 1)  # Down
        elif dx < 0 and dy == 0:
            return (-1, 0)  # Left
        else:
            raise ValueError(f"Invalid direction from {src} to {target}. Only cardinal directions are supported.")