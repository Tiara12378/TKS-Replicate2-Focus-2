import torch
import torch.nn as nn
import numpy as np
from grid_world import GridWorld

# ===== 1. LOAD DATA =====
inputs = np.load("inputs.npy")
targets = np.load("targets.npy")

X = torch.tensor(inputs, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.float32)

# ===== 2. DEFINE MODEL =====
class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = WorldModel()

# ===== 3. TRAINING SETUP =====
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

epochs = 2500

# ===== 4. TRAINING LOOP =====
for epoch in range(epochs):
    optimizer.zero_grad()
    
    predictions = model(X)
    loss = criterion(predictions, y)
    
    loss.backward()
    optimizer.step()

# ===== 5. SINGLE STEP EVALUATION =====
print("\n=== SINGLE STEP EVALUATION (dataset) ===")

with torch.no_grad():
    total_loss = criterion(model(X), y)
    print("Loss on entire dataset:", total_loss.item())

    for i in range(5):
        inp = X[i]
        true = y[i]
        pred = model(inp)
        rounded = torch.round(pred)

        print("Input:", inp)
        print("True:", true)
        print("Pred:", pred)
        print("Rounded:", rounded)
        print("------")

# ===== 6. MULTI STEP SIMULATION =====
print("\n=== PURE MODEL ROLLOUT (IMAGINATION MODE) ===")

env = GridWorld()
real_start = env.reset()

print("Start state:", real_start)

with torch.no_grad():
    
    current_state = real_start
    
    for step in range(10):
        
        action = np.random.randint(0, 4)
        
        # build input from CURRENT (possibly predicted) state
        action_one_hot = [0,0,0,0]
        action_one_hot[action] = 1
        
        model_input = torch.tensor(
            [current_state[0], current_state[1]] + action_one_hot,
            dtype=torch.float32
        )
        
        predicted_next = model(model_input)
        predicted_rounded = torch.round(predicted_next)
        
        # compute what real env WOULD have done (for comparison only)
        real_next = env.step(action)
        
        print(f"Step {step}")
        print("Action:", action)
        print("Real Next:", real_next)
        print("Predicted Next:", predicted_rounded.numpy())
        print("Raw Prediction:", predicted_next.numpy())
        print("------")
        
        # now update state USING MODEL PREDICTION (not real env)
        current_state = (
            int(predicted_rounded[0].item()),
            int(predicted_rounded[1].item())
        )


#==== 7. HEURISTIC-GUIDED PLANNING =====

print("\n=== HEURISTIC-GUIDED PLANNING ===")

env = GridWorld()

goal = (4, 4)
start_state = env.reset()

print("Start:", start_state)
print("Goal:", goal)

best_sequence = None
best_score = float("inf")

with torch.no_grad():
    
    for trial in range(300):
        
        current_state = start_state
        action_sequence = []
        
        for step in range(10):
            
            x, y = current_state
            goal_x, goal_y = goal
            
            # heuristic probabilities
            probs = [0.1, 0.1, 0.1, 0.1]  # base probability
            
            if goal_x > x:
                probs[1] += 0.4  # action 1 = down
            if goal_x < x:
                probs[0] += 0.4  # action 0 = up
            if goal_y > y:
                probs[3] += 0.4  # action 3 = right
            if goal_y < y:
                probs[2] += 0.4  # action 2 = left
            
            probs = np.array(probs)
            probs = probs / probs.sum()
            
            action = np.random.choice([0,1,2,3], p=probs)
            action_sequence.append(action)
            
            action_one_hot = [0,0,0,0]
            action_one_hot[action] = 1
            
            model_input = torch.tensor(
                [current_state[0], current_state[1]] + action_one_hot,
                dtype=torch.float32
            )
            
            predicted_next = model(model_input)
            predicted_rounded = torch.round(predicted_next)
            
            current_state = (
                int(predicted_rounded[0].item()),
                int(predicted_rounded[1].item())
            )
            
            if current_state == goal:
                break
        
        distance = abs(current_state[0] - goal[0]) + abs(current_state[1] - goal[1])
        score = distance + 0.1 * len(action_sequence)
        
        if score < best_score:
            best_score = score
            best_sequence = action_sequence

print("Best sequence found:", best_sequence)
print("Best score:", best_score)