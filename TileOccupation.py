class TileOccupation:
    def __init__(self, tile, turns, owner):
        """
        Represents the occupation of a tile.
        :param tile: Tuple (x, y) representing the tile's position.
        :param turns: List of turns during which the tile is occupied.
        :param owner: The unit ID of the robot occupying the tile.
        """
        self.tile = tile
        self.turns = turns  # List of turns
        self.owner = owner
    
    def is_tile_occupied(self, tile, turn, unit_id=None):
        """
        Checks if a tile is occupied during a specific turn.
        :param tile: Tuple (x, y) representing the tile's position.
        :param turn: The turn to check.
        :param unit_id: The unit ID of the robot trying to access the tile.
        :return: True if the tile is occupied by another unit, False otherwise.
        """
        for occupation in self.tile_occupation:
            if occupation.tile == tile and turn in occupation.turns:
                # Allow access if the tile is occupied by the same unit
                if unit_id and occupation.owner == unit_id:
                    return False
                return True
        return False