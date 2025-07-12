#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Define matrix A (diagonal) and matrix P (change of basis)
A = np.array([[2, 0],
              [0, 3]])

# P will be a matrix with some invertible columns (not necessarily eigenvectors here)
P = np.array([[1, 1],
              [1, -1]])

# Calculate B = P^{-1} A P
P_inv = np.linalg.inv(P)
print(f"P_inv: {P_inv}")

B = P_inv @ A @ P
print(f"B: {B}")

# Define a test vector x (arbitrary)
x = np.array([[1],
              [2]])

# Transform x using A in the original basis
Ax = A @ x

# Transform x using B in the new basis
# First change x into new basis via P^{-1}, then apply B, then go back via P
x_new_basis = P_inv @ x
Bx_new_basis = B @ x_new_basis
Bx = P @ Bx_new_basis

# Prepare plot
fig, ax = plt.subplots()
origin = np.zeros(2)

ax.quiver(*origin, *x.flatten(), color='gray', angles='xy', scale_units='xy', scale=1, label='original x')
# ax.quiver(*origin, *Ax.flatten(), color='blue', angles='xy', scale_units='xy', scale=1, label='A·x')
ax.quiver(*origin, *Bx.flatten(), color='red', angles='xy', scale_units='xy', scale=1, label='PBP^{-1}·x (via similarity)')

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
plt.title("Transformation by A vs. B (via similarity)")
plt.show()

Ax, Bx  # return numerical results too
