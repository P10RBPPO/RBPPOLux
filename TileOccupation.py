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
    
    