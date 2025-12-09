# Monkey AGI (Reinforcement Learning)

A minimal reinforcement-learning agent that learns to choose between two identical taps:
one contains sugar water (reward = +5) and the other plain water (reward = 0).
The agent has no built-in rules and learns purely from trial and error using an
Advantage Actor–Critic (A2C) neural network.

This project demonstrates:
- autonomous learning from environment reward,
- policy gradient reinforcement learning,
- visualization of behaviour,
- adaptation when environment reward changes (tap switch).

---

## Features
- **RL Environment** (`TapEnv`) — stateless 2-armed bandit with episodic structure.
- **Actor–Critic Agent** — neural network with policy + value head.
- **Training Loop** — standard A2C update (no hand-coded behaviours).
- **Visualisation** — matplotlib scenes showing:
  - the taps (left=red, right=blue),
  - the Monkey’s chosen action,
  - probability of choosing left over time,
  - rolling reward curve.
- **Environment switches** to test adaptation (optional).

---

## Installation

```bash
git clone https://github.com/aelamin25/monkey-agi-rl.git
cd monkey-agi-rl
pip install -r requirements.txt
