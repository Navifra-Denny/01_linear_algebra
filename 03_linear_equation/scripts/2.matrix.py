#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 벡터 정의
a1 = np.array([1, -4, -3])
a2 = np.array([3, 2, -2])
a3 = np.array([4, -6, -7])

# 평면 생성: Span{a1, a2}
s = np.linspace(-2, 2, 20)
t = np.linspace(-2, 2, 20)
S, T = np.meshgrid(s, t)

X = S * a1[0] + T * a2[0]
Y = S * a1[1] + T * a2[1]
Z = S * a1[2] + T * a2[2]

# 시각화
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 평면
plane = ax.plot_surface(X, Y, Z, alpha=0.5, color='lightblue', edgecolor='gray', linewidth=0.2)
ax.text(5, -15, -8, 'Span{a1, a2}', color='navy', fontsize=10, weight='bold')

# 벡터들
ax.quiver(0, 0, 0, *a1, color='r', label='a1')
ax.quiver(0, 0, 0, *a2, color='g', label='a2')
ax.quiver(0, 0, 0, *a3, color='b', label='a3')

# 설정
ax.set_title('Plane spanned by a1 and a2 (Span{a1, a2})')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()

plt.tight_layout()
plt.show()
