import random

class Memory:
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.paths = []

    def add_paths(self, paths):
        self.paths.extend(paths)
        self._trim_memory()

    def _trim_memory(self):
        if len(self.paths) > self.max_capacity:
            num_paths_to_remove = len(self.paths) - self.max_capacity
            self.paths = self.paths[num_paths_to_remove:]

    def get_paths(self):
        return self.paths

    def clear_memory(self):
        self.paths = []
        
        

def select_random_paths(paths, percentage, seed=None):
    if seed is not None:
        random.seed(seed)
    num_to_select = int(len(paths) * percentage)
    return random.sample(paths, num_to_select)