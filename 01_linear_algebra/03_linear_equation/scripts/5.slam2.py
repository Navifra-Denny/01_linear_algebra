#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 상태변수: [x1x, x1y, x2x, x2y, lx, ly]
# 이동: x2 = x1 + u,     u = [1.0, 0.5]
# 관측1: l - x1 = z1,    z1 = [2.0, 1.5]
# 관측2: l - x2 = z2,    z2 = [1.0, 1.0]

# Ax = b 구성
A = np.array([
    [-1,  0, 1,  0, 0, 0],  # x2x - x1x = u_x
    [ 0, -1, 0,  1, 0, 0],  # x2y - x1y = u_y
    [-1,  0, 0,  0, 1, 0],  # lx - x1x = z1x
    [ 0, -1, 0,  0, 0, 1],  # ly - x1y = z1y
    [ 0,  0, -1, 0, 1, 0],  # lx - x2x = z2x
    [ 0,  0, 0, -1, 0, 1],  # ly - x2y = z2y
])

b = np.array([
    1.0,   # u_x
    0.5,   # u_y
    2.0,   # z1x
    1.5,   # z1y
    1.0,   # z2x
    1.0    # z2y
])

# 최소자승 해 구하기
x_est = np.linalg.lstsq(A, b, rcond=None)[0]

# 결과 정리 및 출력
labels = ['x1x (초기 x)', 'x1y (초기 y)', 'x2x (이동 후 x)', 'x2y (이동 후 y)', 'lx (랜드마크 x)', 'ly (랜드마크 y)']
df = pd.DataFrame({'변수': labels, '추정값': np.round(x_est, 4)})

# 표 출력
print(df.to_string(index=False))

# 시각화 (x1, x2, landmark)
x1 = x_est[0:2]
x2 = x_est[2:4]
landmark = x_est[4:6]

plt.figure(figsize=(6, 6))
plt.plot([x1[0], x2[0]], [x1[1], x2[1]], 'o-k', label='로봇 경로')
plt.scatter(*landmark, c='red', marker='x', label='랜드마크')
plt.text(*x1, 'x1', fontsize=12, ha='right')
plt.text(*x2, 'x2', fontsize=12, ha='right')
plt.text(*landmark, 'l', fontsize=12, ha='right')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title('2D SLAM 추정 결과')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
