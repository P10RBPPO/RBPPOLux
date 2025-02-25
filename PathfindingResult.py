class PathfindingResult:
    def __init__(self, path, move_cost, action_queue):
        self.path = path
        self.move_cost = move_cost
        self.action_queue = action_queue