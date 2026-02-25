import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = None
        
        # define unsafe cells
        self.unsafe_cells = [(1,1), (2,3), (4,0)]
        
    def reset(self):
        # start agent in random safe cell
        while True:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            if (x,y) not in self.unsafe_cells:
                self.state = (x,y)
                return self.state
    
    def step(self, action):
        x, y = self.state
        
        if action == 0:  # up
            x = max(0, x-1)
        elif action == 1:  # down
            x = min(self.size-1, x+1)
        elif action == 2:  # left
            y = max(0, y-1)
        elif action == 3:  # right
            y = min(self.size-1, y+1)
        
        next_state = (x,y)
        self.state = next_state
        
        return next_state