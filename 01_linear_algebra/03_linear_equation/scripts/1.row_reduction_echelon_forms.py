#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 매개변수 범위 설정
s_vals = np.linspace(-2, 2, 10)
t_vals = np.linspace(-2, 2, 10)

# 격자 생성
S, T = np.meshgrid(s_vals, t_vals)

# 일반해 표현식: x = s*v1 + t*v2 + p
v1 = np.array([-6, 1, 0, 0, 0])
v2 = np.array([-3, 0, 4, 1, 0])
p =  np.array([0, 0, 5, 0, 7])

# 각 성분 계산 (x1, x2, x3만 시각화)
X1 = -6*S - 3*T
X2 = S
X3 = 5 + 4*T

# 3D 시각화
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# 포인트 플롯
ax.plot_surface(X1, X2, X3, alpha=0.6, cmap='viridis', edgecolor='gray')

# 라벨 설정
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('General Solution Surface of the System (x1, x2, x3)')

plt.tight_layout()
plt.show()
