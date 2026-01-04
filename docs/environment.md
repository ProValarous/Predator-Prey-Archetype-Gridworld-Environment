
# GridWorld Environment

## Overview

The `GridWorldEnv` is the core simulation arena where predator and prey agents
interact. It implements a discrete, bounded 2D grid where agents move, collide,
and capture each other according to well-defined rules.

This page explains the **concepts**. For method signatures and parameters, see the
[API Reference](api/gridworld.md).

---

## Coordinate System

The environment uses a standard **Cartesian coordinate system**:

| Property | Value |
|----------|-------|
| **Origin** | Bottom-left corner `(0, 0)` |
| **X-axis** | Increases rightward |
| **Y-axis** | Increases upward |
| **Bounds** | `[0, size-1]` for both axes |