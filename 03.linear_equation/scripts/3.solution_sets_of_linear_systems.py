#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 예제: Ax = 0 형태의 선형 시스템
# A는 2x3 행렬 (자유변수 1개 → 해가 무한대)
A = np.array([
    [1, 2, -1],
    [0, 1,  2]
])

# null space 구하기: Ax = 0을 만족하는 x들
# scipy를 사용하지 않고 해 공간을 수작업으로 구성 (단순한 예제이므로)
# rref 결과로부터 x3 = t 를 자유변수로 두면:
# x1 = -2x2 + x3 → x1 = -2s + t
# x2 = s, x3 = t
s_vals = np.linspace(-2, 2, 20)
t_vals = np.linspace(-2, 2, 20)
S, T = np.meshgrid(s_vals, t_vals)

X1 = -2 * S + T
X2 = S
X3 = T

# 다시 그림을 그리되, 원점에 좌표축 교차점을 시각적으로 표시
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 해 공간 (null space) 평면
ax.plot_surface(X1, X2, X3, alpha=0.6, color='lightgreen', edgecolor='gray', linewidth=0.2)

# 원점 표시
ax.scatter(0, 0, 0, color='red', s=50, label='origin (0,0,0)')
ax.text(0, 0, 0.5, 'origin', color='red')

# 축 설정 및 라벨
ax.set_title('Solution set of Ax = 0 (Null space)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.legend()

plt.tight_layout()
plt.show()