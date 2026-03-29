# Model-based-planner-decision-system

Can an AI learn to imagine consequences before acting — and use that imagination to stay safe?
This project explores a foundational question in AI safety: what happens when you give an agent the ability to simulate its own future before committing to a decision? Built in a 5×5 GridWorld, this system implements a learned neural world model, imagination-based rollouts, and a three-strategy planner comparison experiment across 50 episodes each.
The core finding: safety structured intelligence, rather than limiting it.
What This Is
This is TKS Focus Project 2 — a consequence-aware, model-based AI planning system. The agent doesn't just react to its environment. It builds an internal model of how the world works, imagines the outcomes of different action sequences, and selects paths that balance goal-seeking with danger avoidance.
This connects to a broader research question: how do we build AI systems that reason about consequences the way a cautious, curious mind does — before it's too late to course-correct?
Architecture Overview
GridWorld Environment
        ↓
  Data Collection (collect_data.py)
  1000 (state, action) → next_state transitions
        ↓
  Neural World Model (world_model.py)
  Input: [x, y, a0, a1, a2, a3]  →  Output: [next_x, next_y]
  Architecture: Linear(6→32) → ReLU → Linear(32→2)
  Trained with MSELoss, Adam optimizer, 2500 epochs
        ↓
  Imagination Rollouts
  Agent simulates 10-step futures using the world model
  (without touching the real environment)
        ↓
  Planner Comparison Experiment
  Three strategies × 50 episodes each
The Experiment
Three planners were compared head-to-head across 50 episodes, all navigating toward goal (4,4) while a danger zone sits at (2,2):
Planner
Strategy
Goal-Seeking
Safety Awareness
Random
Uniform random action selection
✗
✗
Heuristic
Probability-weighted toward goal
✓
✗
Safe
Heuristic + danger direction avoidance
✓
✓
The safe planner suppresses action probabilities leading toward the danger zone while preserving goal-directed behavior — demonstrating that constraint-aware planning and effective navigation are not in conflict.
Repository Structure
TKS-Replicate2-Focus-2/
├── grid_world.py        # 5×5 GridWorld environment with unsafe cells
├── collect_data.py      # Data collection: 1000 random transitions
├── world_model.py       # Neural world model + all planning experiments
└── test_env.py          # Environment sanity checks
grid_world.py
Defines the GridWorld class: a 5×5 grid with boundary-enforced movement (up/down/left/right) and three hardcoded unsafe cells at (1,1), (2,3), and (4,0). The agent always starts in a randomly selected safe cell.
collect_data.py
Generates the training dataset by running a random policy for 1000 steps, resetting every 50 steps for diversity. Each sample is a 6-dimensional input vector [x, y, a0, a1, a2, a3] paired with the resulting [next_x, next_y] target.
world_model.py
The core of the project. Contains:
Model training — 2500-epoch Adam optimization with MSELoss
Single-step evaluation — verifies prediction accuracy across the dataset
Pure model rollout (imagination mode) — 10-step simulation using only model predictions, no real environment queries
Heuristic-guided planning with safety — 700-trial best-sequence search with danger zone penalty
Planner comparison experiment — 50-episode benchmark across random, heuristic, and safe planners
Key Concepts
World Model: A small neural network trained to predict next_state given (state, action). Once trained, it acts as an internal simulator — the agent can query it thousands of times in imagination without ever touching the real environment.
Imagination-Based Planning: Before taking any action, the agent simulates multiple future trajectories using its world model. This is the core mechanism behind model-based RL systems like Dreamer, MuZero, and MBPO — explored here at minimal scale to isolate the foundational idea.
Safety-Penalized Planning: The safe planner doesn't just avoid danger reactively. It steers away from dangerous directions during action selection, encoding safety as a prior over behavior rather than a post-hoc filter.
Setup & Usage
Requirements
Python 3.8+
numpy
torch
Install dependencies:
pip install numpy torch
Run the pipeline:
# Step 1: Generate training data
python collect_data.py

# Step 2: Train the world model and run all experiments
python world_model.py
Expected outputs:
Loss on training dataset
10-step imagination rollout vs. real environment comparison
Best action sequence from heuristic-guided planning
Planner comparison: success rate, average steps to goal, danger zone passes per strategy
Why This Matters
The gap between an AI that reacts and one that anticipates is the gap between brittle and robust. A model-based agent that imagines consequences before committing is — at least structurally — a step toward the kind of careful, consequence-aware reasoning we need from systems operating in high-stakes environments.
This project is a minimal proof-of-concept for that structure. The GridWorld is small by design. The question it's asking isn't.
Bigger Picture
This project sits at the intersection of:
Model-based reinforcement learning — learned world models, planning in imagination
AI safety — constraint-aware planning, danger avoidance without sacrificing capability
Intrinsic motivation / curiosity-driven learning — future direction: replacing the heuristic with learned uncertainty or novelty signals

- Built as part of TKS (The Knowledge Society) Focus Project 2.
