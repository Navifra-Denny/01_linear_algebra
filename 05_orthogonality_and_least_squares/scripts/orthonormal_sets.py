#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define orthonormal basis U
u1 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
u2 = np.array([2/3, -2/3, 1/3])
U = np.column_stack((u1, u2))  # U is 3x2

# Define x in B-coordinate (U basis)
x_B = np.array([np.sqrt(2), 3])  # Coordinates in the basis U

# Convert to standard coordinates
x_std = U @ x_B  # x in standard coordinates

# Prepare vectors for visualization
origin = np.zeros(3)
vectors = {
    "U basis u1": u1,
    "U basis u2": u2,
    "x (in U coords)": x_std,
}

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw vectors
colors = ['r', 'g', 'b']
for i, (name, vec) in enumerate(vectors.items()):
    ax.quiver(*origin, *vec, color=colors[i], label=name)

# Set axes limits
ax.set_xlim([-1, 4])
ax.set_ylim([-2, 2])
ax.set_zlim([-1, 2])

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Visualization of U basis vectors and x = Ux")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

