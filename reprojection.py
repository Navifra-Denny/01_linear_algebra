#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
# matplotlib.use('QtAgg')
print(matplotlib.get_backend())
# QtAgg
# 1. 원본 벡터 x (2차원 예제)
x = np.array([2, 3])

# 2. 정사영할 방향 e (정규화된 단위벡터)
e = np.array([1, 0.5])
e = e / np.linalg.norm(e)  # 정규화 (단위벡터로 만들기)

print("2. 정사영할 방향 e")
print(f"matrix e: {e}")
print("------------------")
# 3. 정사영 행렬 P = e @ e.T
# P = np.outer(e, e)  # e e^T
P = np.array([[e[0]*e[0], e[0]*e[1]],
              [e[1]*e[0], e[1]*e[1]]])

# 4. 정사영된 벡터
x_proj = P @ x

# 5. 시각화
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, color='blue', label='x (original)')
plt.quiver(0, 0, x_proj[0], x_proj[1], angles='xy', scale_units='xy', scale=1, color='red', label='Projection EET x')
plt.plot([0, e[0]*3], [0, e[1]*3], 'g--', label='E direction')
# plt.plot([0, e[0]*3], [0, e[1]*3], 'g--', label='E direction')
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.gca().set_aspect('equal')
plt.grid(True)
plt.legend()
plt.title("Projection of x onto direction e via P = eeᵀ")
plt.show()
