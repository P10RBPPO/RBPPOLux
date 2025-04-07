import heapq
import math
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys
from PathfindingResult import PathfindingResult
from TileOccupation import TileOccupation

class RobotController:
    def __init__(self, game_state, player):
        self.game_state = game_state
        self.player = player
        self.units = {}
        self.unit_types = {}
        self.unit_roles = {}
        self.claimed_ice_tiles = {}  # Tracks which ice tiles are claimed by which robot
        self.robot_to_factory = {}  # Tracks which factory each robot is assigned to
        self.occupied_tiles = set()  # Set of currently occupied tiles

    def add_unit(self, unit_id, unit, unit_type):
        """
        Adds a new unit to the controller and assigns it a role and factory.
        """
        self.units[unit_id] = unit
        self.unit_types[unit_id] = unit_type

        # Automatically assign the robot to the closest factory
        factory_tiles, _ = self.get_factories(self.game_state)
        if len(factory_tiles) > 0:
            factory_distances = np.linalg.norm(factory_tiles - unit.pos, axis=1)
            closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
            self.assign_factory(unit_id, closest_factory_tile)

        # Assign a role using the determine_role method
        if unit_id not in self.unit_roles:
            role = self.determine_role(unit_id, unit)
            self.assign_role(unit_id, role)
            print(f"Assigned role '{role}' to unit {unit_id}", file=sys.stderr)

    def assign_role(self, unit_id, role):
        self.unit_roles[unit_id] = role
    
    def assign_factory(self, unit_id, factory_pos):
        self.robot_to_factory[unit_id] = factory_pos

    def determine_role(self, unit_id, unit):
        """
        Determines the role of a unit based on its index in the self.units dictionary.
        Alternates between "Ore Miner" and "Ice Miner".
        """
        # Get the index of the unit in the self.units dictionary
        unit_index = list(self.units.keys()).index(unit_id)

        # Alternate roles based on the index
        if unit_index % 2 == 0:
            return "Ore Miner"
        else:
            return "Ice Miner"
    
    def update_game_state(self, new_game_state):
        """
        Updates the game state and synchronizes all units.
        Reassigns robots to new factories if their assigned factory is dead.
Frees claimed tiles for dead robots.
        """
        self.game_state = new_game_state

        # Get the current unit IDs from the new game_state
        current_unit_ids = set(new_game_state.units[self.player].keys())

        # Remove dead units
        self.units = {unit_id: unit for unit_id, unit in self.units.items() if unit_id in current_unit_ids}
        self.unit_types = {unit_id: unit_type for unit_id, unit_type in self.unit_types.items() if unit_id in current_unit_ids}
        self.unit_roles = {unit_id: role for unit_id, role in self.unit_roles.items() if unit_id in current_unit_ids}
        self.robot_to_factory = {unit_id: factory for unit_id, factory in self.robot_to_factory.items() if unit_id in current_unit_ids}

        # Free claimed tiles for dead robots
        self.claimed_ice_tiles = {tile: claimant for tile, claimant in self.claimed_ice_tiles.items() if claimant in current_unit_ids}

        # Get the current factory IDs and positions
        current_factories = new_game_state.factories[self.player]
        current_factory_positions = {tuple(factory.pos): factory for factory in current_factories.values()}

        # Reassign robots if their assigned factory is dead
        for unit_id, assigned_factory in list(self.robot_to_factory.items()):
            if tuple(assigned_factory) not in current_factory_positions:
                # The assigned factory is dead, reassign to the closest factory
                factory_tiles, _ = self.get_factories(new_game_state)
                if len(factory_tiles) > 0:
                    factory_distances = np.linalg.norm(factory_tiles - self.units[unit_id].pos, axis=1)
                    closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                    self.assign_factory(unit_id, closest_factory_tile)
                    print(f"Reassigned robot {unit_id} to new factory at {closest_factory_tile}", file=sys.stderr)
                else:
                    # No factories left, unassign the robot
                    del self.robot_to_factory[unit_id]
                    print(f"Unassigned robot {unit_id} because no factories are left", file=sys.stderr)

        # Update existing units and add new units
        for unit_id, unit in new_game_state.units[self.player].items():
            if unit_id in self.units:
                # Update the state of existing units
                self.units[unit_id] = unit
            else:
                # Add new units
                self.add_unit(unit_id, unit, unit.unit_type)

    def control_units(self, actions, game_state):
        """
        Controls all units and updates the actions dictionary.
        """
        self.update_game_state(game_state)  # Update the game state
        # Clear and repopulate the occupied tiles set
        self.occupied_tiles = {tuple(unit.pos) for unit in self.units.values()}

        ice_tile_locations = self.get_ice_tile_locations(self.game_state)
        ore_tile_locations = self.get_ore_tile_locations(self.game_state)

        for unit_id, unit in self.units.items():
            if len(unit.action_queue) == 0:
                role = self.unit_roles.get(unit_id, None)
                if role == "Ice Miner":
                    actions[unit_id] = self.control_ice_miner(unit_id, unit, ice_tile_locations)
                elif role == "Ore Miner":
                    actions[unit_id] = self.control_ore_miner(unit_id, unit, ore_tile_locations)
                else:
                    actions = None
                    print(f"Warning: Unit {unit_id} has no role assigned.", file=sys.stderr)

        # Resolve conflicts before returning actions
        actions = self.resolve_conflicts(actions, game_state)
        return actions

    def control_ice_miner(self, unit_id, unit, ice_tile_locations):
        # Get the assigned factory for this robot
        assigned_factory = self.robot_to_factory.get(unit_id, None)

        # If no factory is assigned, skip (this should not happen if add_unit is used correctly)
        if assigned_factory is None:
            print(f"Warning: Unit {unit_id} has no assigned factory.", file=sys.stderr)
            return

        # If the robot is carrying ice, return to the assigned factory
        if unit.cargo.ice >= 60:
            return self.return_to_factory(unit_id, unit, assigned_factory, resource_type=0, resource_amount=unit.cargo.ice)
        # If the robot is not carrying ice, go to the closest unclaimed ice tile
        elif unit.cargo.ice < 60:
            closest_ice_tile = self.claim_tile(unit_id, unit, ice_tile_locations, self.claimed_ice_tiles, self.game_state.real_env_steps)
            if closest_ice_tile is not None:
                if np.all(closest_ice_tile == unit.pos):  # If the robot is already on the ice tile
                    if unit.power >= unit.dig_cost(self.game_state) + unit.action_queue_cost(self.game_state):
                        return [unit.dig(repeat=0, n=1)]  # Perform the digging action
                    else:
                        return [unit.recharge(x=unit.dig_cost(self.game_state) + unit.action_queue_cost(self.game_state))]
                else:
                    #print(f"Unit {unit_id} moving to ice tile {closest_ice_tile}", file=sys.stderr)
                    return self.move_to_tile(unit, closest_ice_tile, self.game_state.real_env_steps)

    def control_ore_miner(self, unit_id, unit, ore_tile_locations):
        # Get the assigned factory for this robot
        assigned_factory = self.robot_to_factory.get(unit_id, None)

        # If no factory is assigned, skip (this should not happen if add_unit is used correctly)
        if assigned_factory is None:
            print(f"Warning: Unit {unit_id} has no assigned factory.", file=sys.stderr)
            return

        # If the robot is carrying ore, return to the assigned factory
        if unit.cargo.ore >= 60:
            return self.return_to_factory(unit_id, unit, assigned_factory, resource_type=1, resource_amount=unit.cargo.ore)
        # If the robot is not carrying ore, go to the closest unclaimed ore tile
        elif unit.cargo.ore < 60:
            closest_ore_tile = self.claim_tile(unit_id, unit, ore_tile_locations, self.claimed_ice_tiles, self.game_state.real_env_steps)

            if closest_ore_tile is not None:
                if np.all(closest_ore_tile == unit.pos):  # If the robot is already on the ore tile
                    if unit.power >= unit.dig_cost(self.game_state) + unit.action_queue_cost(self.game_state):
                        return [unit.dig(repeat=0, n=1)]  # Perform the digging action
                    else:
                        return [unit.recharge(x=unit.dig_cost(self.game_state) + unit.action_queue_cost(self.game_state))]
                else:
                    #print(f"Unit {unit_id} moving to ore tile {closest_ore_tile}", file=sys.stderr)
                    return self.move_to_tile(unit, closest_ore_tile, self.game_state.real_env_steps)

    
    def resolve_conflicts(self, actions, game_state):
        """
        Resolves conflicts by checking if two robots are about to move into the same tile
        or if a robot is trying to move into a tile already occupied by another robot.
        This includes both newly decided actions and actions already in progress.
        """
        direction_map = {
            0: (0, 0),   # center (no movement)
            1: (0, -1),  # up
            2: (1, 0),   # right
            3: (0, 1),   # down
            4: (-1, 0)   # left
        }

        target_positions = {}  # Maps target positions to the list of robots intending to move there
        resolved_actions = {}  # Final resolved actions for each robot

        # Track current positions of all robots
        current_positions = {unit_id: tuple(unit.pos) for unit_id, unit in game_state.units[self.player].items()}

        # First pass: Analyze new actions and populate target_positions
        # print(f"Turn {game_state.real_env_steps} size of actions: {len(actions)}", file=sys.stderr)
        for unit_id, action_list in actions.items():
            if len(action_list) > 0:  # Explicitly check if the action list is not empty
                current_pos = current_positions[unit_id]
                action = action_list[0]  # Only consider the first action in the queue
                if isinstance(action, np.ndarray) and action[0] == 0:  # Check if it's a move action
                    direction = action[1]
                    dx, dy = direction_map[direction]
                    target_pos = (current_pos[0] + dx, current_pos[1] + dy)

                    # Check if the target position is already occupied
                    if target_pos in current_positions.values():
                        #print(f"Turn {game_state.real_env_steps}: Robot {unit_id} cannot move to {target_pos} because it is occupied.", file=sys.stderr)
                        resolved_actions[unit_id] = []  # Cancel the movement action
                    else:
                        # Add the robot to the target_positions map
                        if target_pos not in target_positions:
                            target_positions[target_pos] = []
                        target_positions[target_pos].append(unit_id)
                else:
                    # If it's not a move action, keep it as is
                    resolved_actions[unit_id] = action_list

        # Second pass: Check action queues from game_state for conflicts
        for unit_id, unit in game_state.units[self.player].items():
            if len(unit.action_queue) > 0:  # Check if the unit has an action queue
                current_pos = tuple(unit.pos)
                action = unit.action_queue[0]  # Only consider the first action in the queue
                if isinstance(action, np.ndarray) and action[0] == 0:  # Check if it's a move action
                    direction = action[1]
                    #reverse_direction = self.reverse_direction(direction)
                    dx, dy = direction_map[direction]
                    target_pos = (current_pos[0] + dx, current_pos[1] + dy)

                    # Check if the target position is already occupied or has a conflict
                    if target_pos in current_positions.values():
                        #print(f"Turn {game_state.real_env_steps}: Robot {unit_id} cannot move to {target_pos} because it is occupied.", file=sys.stderr)
                        resolved_actions[unit_id] = []
                        #resolved_actions[unit_id] = [unit.move(reverse_direction)]  # Cancel the movement action
                    elif target_pos in target_positions:
                        #print(f"Turn {game_state.real_env_steps}: Conflict detected at {target_pos} for robot {unit_id}.", file=sys.stderr)
                        target_positions[target_pos].append(unit_id)
                    else:
                        # Add the robot to the target_positions map
                        target_positions[target_pos] = [unit_id]

        # Resolve conflicts for shared target positions
        for target_pos, unit_ids in target_positions.items():
            if len(unit_ids) > 1:
                # Conflict detected: Multiple robots want to move to the same tile
                #print(f"Turn {game_state.real_env_steps}: Conflict detected at {target_pos} for robots {unit_ids}.", file=sys.stderr)
                # Sort robots by their IDs to prioritize one robot
                unit_ids.sort(key=lambda uid: int(uid.split('_')[1]))  # Prioritize by robot ID
                for uid in unit_ids[1:]:  # Allow the first robot to proceed, others must wait
                    resolved_actions[uid] = []  # Cancel the movement action
                    #print(f"Turn {game_state.real_env_steps}: Robot {uid} is waiting due to conflict at {target_pos}.", file=sys.stderr)

        # Add non-conflicting actions to resolved_actions
        for unit_id, action_list in actions.items():
            if unit_id not in resolved_actions:
                resolved_actions[unit_id] = action_list

        return resolved_actions

    def claim_tile(self, unit_id, unit, tile_locations, claimed_tiles, current_turn):
        """
        Claims the closest unclaimed tile (e.g., ice or ore) for the given unit or returns the already-claimed tile.
        Marks the tile as occupied from the arrival turn until the robot decides to return to the factory.
        """
        # Check if the robot has already claimed a tile
        claimed_tile = None
        for tile, claimant in claimed_tiles.items():
            if claimant == unit_id:
                claimed_tile = np.array(tile)
                break

        if claimed_tile is not None:
            # If the robot has already claimed a tile, return it
            return claimed_tile

        # Find the closest unclaimed tile
        unclaimed_tiles = [tile for tile in tile_locations if tuple(tile) not in claimed_tiles]
        if len(unclaimed_tiles) > 0:
            # Prioritize tiles based on distance and robot ID
            unclaimed_tiles = sorted(unclaimed_tiles, key=lambda tile: (
                np.linalg.norm(tile - unit.pos),  # Distance to the tile
                int(unit_id.split('_')[1])       # Robot ID as a tiebreaker
            ))
            for tile in unclaimed_tiles:
                # Estimate when the robot will occupy the tile
                pathfinding_result = PathfindingResult.astar_search(unit, unit.pos, tile, self.game_state, current_turn)
                if pathfinding_result:
                    # Claim the tile for this robot
                    claimed_tiles[tuple(tile)] = unit_id

                    return tile

        # If no unclaimed tiles are available, return None
        return None

    def return_to_factory(self, unit_id, unit, assigned_factory, resource_type, resource_amount):
        """
        Handles the logic for returning to the factory and transferring resources.
        Marks the factory tile as occupied until the robot moves away.
        """
        direction = direction_to(unit.pos, assigned_factory)
        current_turn = self.game_state.real_env_steps

        # remove claim from robot
        if tuple(unit.pos) in self.claimed_ice_tiles:
            del self.claimed_ice_tiles[tuple(unit.pos)]

        # If the robot is adjacent to the factory, perform actions
        if np.array_equal(unit.pos, assigned_factory):
            # Transfer resources to the factory
            if unit.power >= unit.action_queue_cost(self.game_state):
                actions = [unit.transfer(direction, resource_type, resource_amount, repeat=0)]
                # Optionally pick up power from the factory
                factory_power = self.get_closest_factory_unit(unit, self.game_state).power
                power_to_pickup = int(factory_power * 0.20)
                actions.append(unit.pickup(4, power_to_pickup, repeat=0, n=1))
                return actions
            else:
                # If not enough power, just pick up power
                factory_power = self.get_closest_factory_unit(unit, self.game_state).power
                power_to_pickup = int(factory_power * 0.20)
                return [unit.pickup(4, power_to_pickup, repeat=0, n=1)]
        else:
            # Move towards the factory
            return self.move_to_tile(unit, assigned_factory, current_turn)

    def move_to_tile(self, unit, target_tile, current_turn):
        """
        Moves the unit to the target tile if it is not occupied.
        If the target tile is occupied, move to the second-to-last tile in the path.
        """
        # Perform pathfinding to the target tile
        pathfinding_result = PathfindingResult.astar_search(unit, unit.pos, target_tile, self.game_state, current_turn)
        if pathfinding_result:

            # Check if the unit has enough power to execute the path
            if unit.power >= pathfinding_result.total_move_cost + unit.action_queue_cost(self.game_state):
                # Update the occupied tiles set
                self.occupied_tiles.add(tuple(target_tile))
                return pathfinding_result.action_queue  # Return the pathfinding action queue
            else:
                closest_factory = self.get_closest_factory(unit, self.game_state)
                if np.array_equal(unit.pos, closest_factory):
                    factory_power = self.get_closest_factory_unit(unit, self.game_state).power
                    power_to_pickup = int(factory_power * 0.20)
                    return [unit.pickup(4, power_to_pickup, repeat=0, n=1)]
                else:
                    # Return a recharge action if the unit does not have enough power
                    return [unit.recharge(x=pathfinding_result.total_move_cost + unit.action_queue_cost(self.game_state))]

        # If no pathfinding result is found, return an empty action queue
        #print(f"Turn {current_turn}: No path found for robot {unit.unit_id} to target tile {target_tile}.", file=sys.stderr)
        return []

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
    
    def get_closest_factory_unit(self, unit, game_state):
        factory_tiles, factory_units = self.get_factories(game_state)
        factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
        closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
        return factory_units[np.argmin(factory_distances)]
    
    def get_closest_factory(self, unit, game_state):
        factory_tiles, _ = self.get_factories(game_state)
        factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
        closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
        return closest_factory_tile

    def is_enemy_factory(self, target_pos):
        """
        Checks if the target position is occupied by an enemy factory.
        """
        factory_occupancy_map = self.game_state.board.factory_occupancy_map
        factory_at_target = factory_occupancy_map[target_pos[0], target_pos[1]]
        return factory_at_target != -1 and factory_at_target not in self.game_state.teams[self.player].factory_strains
    
    def reverse_direction(self, direction):
        """
        Returns the reverse of the given direction.
        """
        reverse_direction_map = {
            0: 0,  # center
            1: 3,  # up → down
            2: 4,  # right → left
            3: 1,  # down → up
            4: 2   # left → right
        }
        return reverse_direction_map[direction]