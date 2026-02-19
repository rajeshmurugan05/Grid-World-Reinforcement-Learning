# Grid World Reinforcement Learning (Q-Learning)

This project implements a model-free Reinforcement Learning agent using the **Q-Learning algorithm** in a stochastic Grid World environment.

It demonstrates how an agent can learn optimal strategies based on rewards and state transitions, even when the environment is not deterministic.

## Objective

The purpose of this project is to develop a Q-Learning agent that:

- Navigates a grid environment
- Learns optimal action values (Q-values)
- Reaches a goal state while avoiding obstacles
- Handles stochastic transitions (actions may not always go as intended)

## Environment Setup

- **Grid Size:** 5×5
- **Initial State:** (1, 0)
- **Goal State:** (4, 4)
- **Special Jump State:** (3, 3)
- **Obstacles:** (3,2), (2,2), (2,3), (2,4)

## Reward Structure

| Outcome | Reward |
|---------|--------|
| Reach goal normally | +10 |
| Reach via special jump | +15 |
| Any other move | 0 |

## Learning Configuration

- Algorithm: Q-Learning  
- Learning Rate (α): 0.2  
- Discount Factor (γ): 0.9  
- Exploration: Epsilon-Greedy  
- ε (exploration probability): 0.3  
- Environment: Stochastic (80% intended move, 10% side moves)

## How It Works

1. The agent starts at the initial state.
2. It selects actions using an ε-greedy policy.
3. Movement has stochastic outcomes (may deviate from intended direction).
4. Q-values are updated using: Q(s, a) ← Q(s, a) + α [ R + γ max Q(s', a') − Q(s, a) ]
5. Training continues until the agent converges to an optimal policy.

## Running the Code

Make sure you have Python installed (Python 3.x).

To run the script:

```bash
python Grid_World.py
