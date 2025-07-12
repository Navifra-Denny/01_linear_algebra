#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Define matrix A and the eigenvector v1
A = np.array([
    [1.8, 0.8],
    [0.2, 1.2]
])
v1 = np.array([4, 1])  # dominant eigenvector direction (for λ=2)

# Normalize v1 for direction comparison
v1_unit = v1 / np.linalg.norm(v1)

# Define initial vector x
x0 = np.array([-0.51, 1.0])

# Number of steps
num_steps = 9
x_history = [x0]
x = x0
for _ in range(num_steps):
    x = A @ x
    x_history.append(x)

x_history = np.array(x_history)

# Normalize each x_k to get direction
directions = x_history / np.linalg.norm(x_history, axis=1, keepdims=True)

# Compute dot product with v1_unit to measure alignment (cosine similarity)
alignment = directions @ v1_unit

# Plot alignment over k
plt.figure(figsize=(8, 5))
plt.plot(range(num_steps + 1), alignment, marker='o')
plt.title("Alignment with Dominant Eigenvector [4, 1] Over Iterations")
plt.xlabel("k (number of multiplications by A)")
plt.ylabel("cos(θ) between x_k and v1")
plt.grid(True)
plt.tight_layout()
plt.show()

# Return last few iterates and their alignment with v1 direction
x_history[-3:], alignment[-3:]
