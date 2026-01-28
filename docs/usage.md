# Usage Guide

This guide covers installation, basic usage, and how to run the training algorithms.

!!! note "Important"
    The `Agent` and `GridWorldEnv` classes are designed to be used as-is.
    You should **not modify** these core classes. Instead, configure them
    through their parameters and use the provided training scripts.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/ProValarous/Predator-Prey-Archetype-Gridworld-Environment
cd Predator-Prey-Archetype-Gridworld-Environment
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Training Algorithms
The repository provides two main training algorithms:

| Algorithm | Description | Location |
|-----------|-------------|----------|
| **IQL** | Independent Q-Learning | `IQL` |
| **CQL** | Central Q-Learning | `CQL` |

!!! warning "Run from src directory" All training commands should be run from the src directory: ```bash cd src```

## Independent Q-Learning (IQL)

IQL trains each agent independently with its own Q-table. Each agent learns without explicit knowledge of other agents' policies.

### Training
```bash
cd src
python -m baselines.IQL.train_iql [OPTIONS]
```

### IQL Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--episodes` | `int` | `15000` | Number of training episodes |
| `--size` | `int` | `2` | Grid size (size × size) |
| `--alpha` | `float` | `0.1` | Learning rate (Q-value update step size) |
| `--gamma` | `float` | `0.9` | Discount factor (importance of future rewards) |
| `--eps-start` | `float` | `1.0` | Initial exploration rate |
| `--eps-end` | `float` | `0.05` | Final exploration rate |
| `--eps-decay` | `float` | `0.9995` | Exploration decay rate per episode |
| `--seed` | `int` | `0` | Random seed for reproducibility |
| `--save-path` | `str` | `baselines/IQL/iql_qs.npz` | Path to save trained Q-tables |
| `--log-dir` | `str` | `baselines/IQL/logs/` | TensorBoard log directory |

### IQL Training Examples 
```bash
# Basic training (small grid, few episodes)
python -m baselines.IQL.train_iql --episodes 5000 --size 5

# Full training with custom parameters
python -m baselines.IQL.train_iql \
    --episodes 20000 \
    --size 8 \
    --alpha 0.1 \
    --gamma 0.99 \
    --eps-start 1.0 \
    --eps-end 0.01 \
    --seed 42

# Quick test run
python -m baselines.IQL.train_iql --episodes 1000 --size 3 --seed 123
```

### IQL Testing
After training, test the learned policy:
```bash
python -m baselines.IQL.test_iql [OPTIONS]
```

### IQL Testing Parameters 

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--file` | `str` | `baselines/IQL/iql_qs.npz` | Path to trained Q-tables |
| `--size` | `int` | `15` | Grid size (should match training) |
| `--episodes` | `int` | `3` | Number of test episodes |
| `--pause` | `float` | `0.05` | Pause between steps (seconds) |
| `--max-steps` | `int` | `100` | Maximum steps per episode |
| `--seed` | `int` | `None` | Random seed |

### IQL Testing Examples
```bash
# Basic testing with visualization
python -m baselines.IQL.test_iql --size 5 --episodes 3

# Test specific Q-table file
python -m baselines.IQL.test_iql \
    --file baselines/IQL/iql_qs.npz \
    --size 8 \
    --episodes 5 \
    --pause 0.1

# Slower visualization for observation
python -m baselines.IQL.test_iql --size 5 --episodes 1 --pause 0.5
```

## Central Q-Learning
CQL uses a centralized Q-table that considers the joint state of all agents. This allows for coordinated behavior but scales exponentially with agents.

### Training 
```bash 
cd src
python -m baselines.CQL.cql_train [OPTIONS]
```

### CQL Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--episodes` | `int` | `40000` | Number of training episodes |
| `--size` | `int` | `7` | Grid size (size × size) |
| `--alpha` | `float` | `0.25` | Learning rate |
| `--gamma` | `float` | `0.95` | Discount factor |
| `--eps-start` | `float` | `1.0` | Initial exploration rate |
| `--eps-end` | `float` | `0.05` | Final exploration rate |
| `--eps-decay` | `float` | `0.99995` | Exploration decay rate |
| `--seed` | `int` | `0` | Random seed |
| `--predators` | `int` | `2` | Number of predator agents |
| `--preys` | `int` | `2` | Number of prey agents |
| `--save-path` | `str` | `CQL` | Directory to save Q-tables |
| `--log-dir` | `str` | `baselines/CQL/logs/` | TensorBoard log directory |
| `--max-table-gb` | `float` | `16.0` | Maximum Q-table memory (GB) |


### CQL Training Examples 
```bash
# Basic testing with visualization
python -m baselines.IQL.test_iql --size 5 --episodes 3

# Test specific Q-table file
python -m baselines.IQL.test_iql \
    --file baselines/IQL/iql_qs.npz \
    --size 8 \
    --episodes 5 \
    --pause 0.1

# Slower visualization for observation
python -m baselines.IQL.test_iql --size 5 --episodes 1 --pause 0.5
```

### CQL Testing 
```bash
python -m baselines.CQL.test_cql [OPTIONS]
```

## Parameters Gidelines 

### Choosing Grid Size

| Size | Cells | Recommended For |
|------|-------|-----------------|
| `3-5` | 9-25 | Quick testing, debugging |
| `6-8` | 36-64 | Standard experiments |
| `10+` | 100+ | Large-scale experiments |

!!! note "warning" 
    "Memory Usage" CQL's Q-table size grows exponentially with grid size and number of agents. For large grids, use IQL or reduce the number of agent

### Choosing Learning Rate (Alpha)

| Value | Effect |
|-------|--------|
| `0.01 - 0.05` | Slow, stable learning |
| `0.1 - 0.2` | Balanced (recommended) |
| `0.3+` | Fast but potentially unstable |

### Choosing Discount Factor (Gamma)


### Choosing Exploration Parameters

| Parameter | Recommended | Effect |
|-----------|-------------|--------|
| `eps-start` | `1.0` | Full exploration initially |
| `eps-end` | `0.01 - 0.1` | Minimal exploration at end |
| `eps-decay` | `0.999 - 0.9999` | Slower decay = more exploration |


## Saved Files 

### IQL Output Files 

| File | Description |
|------|-------------|
| `baselines/IQL/iql_qs.npz` | Trained Q-tables |
| `baselines/IQL/logs/` | TensorBoard logs |

### CQL Output Files 

| File | Description |
|------|-------------|
| `baselines/CQL/*.npz` | Trained Q-tables |
| `baselines/CQL/logs/` | TensorBoard logs |