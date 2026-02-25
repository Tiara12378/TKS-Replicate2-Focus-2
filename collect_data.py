from grid_world import GridWorld
import numpy as np
import random

# 1. create environment
env = GridWorld()

# 2. storage for dataset
inputs = []
targets = []

# 3. number of transitions to collect
num_transitions = 1000

# 4. start environment
state = env.reset()

for step in range(num_transitions):
    
    # randomly reset every 50 steps to increase diversity
    if step % 50 == 0:
        state = env.reset()
    
    # choose random action (0,1,2,3)
    action = random.randint(0, 3)
    
    # take step in environment
    next_state = env.step(action)
    
    # unpack states
    x, y = state
    next_x, next_y = next_state
    
    # store input and target
    inputs.append([x, y, action])
    targets.append([next_x, next_y])
    
    # update current state
    state = next_state

# 5. convert to numpy arrays
inputs = np.array(inputs)
targets = np.array(targets)

print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)