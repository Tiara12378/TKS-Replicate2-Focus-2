# From Prediction to Consequence-Aware Planning
An AI system can accurately predict the future — and still choose a path that leads directly into failure.
This project investigates a simple but underexplored gap in AI systems:

prediction does not imply good decision-making

# Overview
I built a small model-based planning system to study how different decision strategies affect behavior — while keeping the underlying predictive model constant.

The system:

• learns environment dynamics via a neural network (world model)

• simulates future trajectories

• selects actions based on different evaluation strategies

The goal was not performance, but understanding how decision quality emerges.

# Core Question
If an agent can predict outcomes accurately:

**what determines whether it makes good 
decisions?**

# System Design
**Environment**

• 5×5 grid world

deterministic transitions

goal state + designated danger state

**Learned World Model**
A neural network is trained on transition data:

**(state, action) → next_state**

This enables forward simulation without interacting with the environment.

# Planning Strategies
All planners share the same learned dynamics.

They differ only in how they **search and evaluate futures**:

**Random Planning**

• samples action sequences uniformly

• no structural bias

**Heuristic Planning**

• biases sampling toward goal-directed actions

• introduces directional structure into search

**Risk-Aware Planning**

• penalizes trajectories that pass through dangerous states

• introduces tradeoff between efficiency and safety

# Experimental Results 

Planner: Random

Success: 10/50

Avg Steps: 4.6

Danger Passes: 15

Planner: Heuristic

Success: 34/50

Avg Steps: 5.03

Danger Passes: 8

Planner: Risk-Aware

Success: 42/50

Avg Steps: 6.33

Danger Passes: 4

# Key Observations

**1. Search structure dominates performance**

The heuristic planner improves outcomes without improving prediction.

→ better decisions emerge from how futures are explored, not from better models.

**2. Objective design changes behavior**

Risk-aware planning sacrifices efficiency to avoid dangerous states.

→ optimizing for “shortest path” vs “safest path” produces fundamentally different policies.

**3. Prediction is not the bottleneck**

All planners use identical dynamics.

→ failures arise from evaluation and search, not from inaccurate world models.

# Why This Matters
Many real-world systems fail not because they cannot predict outcomes, but because they:

• evaluate the wrong objective

• fail to consider consequences

• or explore futures inefficiently

This applies directly to:

• autonomous systems

• financial decision-making

• safety-critical AI

# Limitations

• small, fully observable environment

• brute-force trajectory sampling

• no learned value function or policy

This is intentionally simplified to isolate **decision-making behavior**.

# Repository Structure

/environment

/model

/planner

/experiments


# Article


# Final Insight

Prediction answers:

*what will happen?*

Planning answers:

*which future is worth choosing?*

Intelligence emerges from the second.









