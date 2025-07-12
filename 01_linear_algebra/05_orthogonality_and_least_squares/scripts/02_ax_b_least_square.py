#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Least squares example: overdetermined system Ax ≈ b
A = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
])  # A is 3x2, Col(A) ⊂ ℝ³

b = np.array([1, 2, 2])  # b ∈ ℝ³

# Compute least squares solution: x̂ = (AᵀA)⁻¹Aᵀb
x_hat = np.linalg.inv(A.T @ A) @ A.T @ b

# Compute projection of b onto Col(A): ȳ = A x̂
b_proj = A @ x_hat

# Compute residual (b - b_proj)
residual = b - b_proj

# Plotting the vectors in ℝ³
origin = np.zeros(3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Axes for visualization
ax.quiver(*origin, *b, color='blue', label='b (original)', linewidth=2)
ax.quiver(*origin, *b_proj, color='green', label='proj_Col(A)(b)', linewidth=2)
ax.quiver(*b_proj, *(b - b_proj), color='red', label='residual (b - proj)', linestyle='dashed')

ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_zlim([0, 3])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Least Squares: b, its projection onto Col(A), and residual")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
