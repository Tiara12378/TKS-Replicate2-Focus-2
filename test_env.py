from grid_world import GridWorld
import random

env = GridWorld()
state = env.reset()
print("Start:", state)

for i in range(10):
    action = random.randint(0,3)
    next_state = env.step(action)
    print("Action:", action, "Next:", next_state)