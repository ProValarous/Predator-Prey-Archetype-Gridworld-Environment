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

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### 5. Verify Setup 
```bash
# Run tests
pytest tests/ -v # not implemented yet

# Check formatting
black --check src/

# Check linting
flake8 src/
```

## Development Workflow

### Step 1: Sync with Upstream
Before starting any work, ensure your fork is up to date

### Step 2: Create Feature/Fix Branch
Never work directly on `main`. Always create a new branch:
```bash 
git checkout -b feat/short-desc
```
or 
```bash 
git checkout -b fix/short-desc
```

### Step 3: Make Your Changes
Edit the code, add tests, update documentation as needed.

### Step 4: Format and Lint
Before committing, ensure your code meets our standards:
```bash
# Format code with Black
black src/

# Check for linting errors
flake8 src/

# Run tests
pytest tests/ -v
```

### Step 5: Commit Your Changes 
```bash 
git add .
git commit -m "feat: Add your feature description"
```

### Step 6: Push to your fork
```bash
git push -u origin feat/short-desc
```

### Step 7: Open a Pull Request

1. Go to the original repository on GitHub
2. Click "Compare & pull request"
3. Fill in the PR template with a clear description
4. Link any related issues
5. Submit the PR


## Quick Reference 
```bash 
# Complete workflow in one place

# 1. Setup (one-time)
git clone https://github.com/YOUR_USERNAME/Predator-Prey-Gridworld-Environment.git
cd Predator-Prey-Gridworld-Environment
git remote add upstream https://github.com/ProValarous/Predator-Prey-Gridworld-Environment.git
pip install -r requirements-dev.txt
pip install -e .
pre-commit install

# 2. Start new feature
git checkout main
git pull upstream main
git checkout -b feat/my-feature

# 3. Make changes, then format and test
black src/
flake8 src/
pytest tests/ -v

# 4. Commit and push
git add .
git commit -m "feat: Add my feature"
git push -u origin feat/my-feature

# 5. Open PR on GitHub
```