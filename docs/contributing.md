# Contributing to Predator-Prey Gridworld Environment

Thank you for your interest in contributing! This project aims to provide a clean, interpretable environment for Multi-Agent Reinforcement Learning research, and contributions from the community help make that possible.

We welcome contributions of all kinds: bug fixes, new features, documentation improvements, and more. This guide will walk you through the process step-by-step.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all contributors with respect and help us maintain a welcoming community.

---

## Getting Started

### 1. Fork the Repository

Click the **"Fork"** button on the top right of the repository page to create your own copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/Predator-Prey-Gridworld-Environment.git
cd Predator-Prey-Gridworld-Environment
```

### 3. Add Upstream Remote
```bash
git remote add upstream https://github.com/ProValarous/Predator-Prey-Gridworld-Environment.git
```

### 4. Set Up Development Environment
```bash 
# Create virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\Activate.ps1

# Activate it (macOS/Linux)
source venv/bin/activate

# Install the package + development dependencies (editable)
pip install -e ".[dev]"
```

`pip install -e .` makes `multi_agent_package` and `baselines` importable without
setting `PYTHONPATH`, so the verification commands below run as-is from the
repository root.

### 5. Verify Setup 
```bash
# Run tests
python -m pytest tests/ -q

# Check formatting
black --check .

# Check linting
flake8 .
pylint src
```

## Development Workflow

Branching, commit conventions, what CI checks on your PR, and how to
resolve conflicts are all covered in one place: **[Git Workflow](git-workflow.md)**.

The short version: branch from `STRP` (not `main`), never touch `core/`,
run the checks above before pushing, and open your PR against `STRP`.