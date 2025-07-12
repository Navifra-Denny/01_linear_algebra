#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# 시뮬레이션 설정
np.random.seed(0)
n_steps = 50

# 시스템 모델: x_k = x_{k-1} + w
A = 1.0   # 상태 전이
H = 1.0   # 관측 모델
Q = 1e-5  # 프로세스 노이즈 공분산
R = 0.1**2  # 관측 노이즈 공분산

# 실제 상태값과 관측값 생성
x_true = np.zeros(n_steps)
z_meas = np.zeros(n_steps)
x_true[0] = 0.0
for k in range(1, n_steps):
    x_true[k] = A * x_true[k-1] + np.random.normal(0, np.sqrt(Q))
z_meas = x_true + np.random.normal(0, np.sqrt(R), size=n_steps)

# Kalman Filter 초기화
x_est = np.zeros(n_steps)
P = 1.0  # 초기 오차 공분산
x_est[0] = 0.0  # 초기 추정값

for k in range(1, n_steps):
    # 예측 단계
    x_pred = A * x_est[k-1]
    P_pred = A * P * A + Q

    # 갱신 단계
    K = P_pred * H / (H * P_pred * H + R)  # 칼만 이득
    x_est[k] = x_pred + K * (z_meas[k] - H * x_pred)
    P = (1 - K * H) * P_pred

# 시각화를 통해 Kalman Filter 결과를 확인

plt.figure(figsize=(12, 6))
plt.plot(x_true, label="True State", linestyle='--')
plt.plot(z_meas, label="Measured z", alpha=0.5)
plt.plot(x_est, label="Estimated x", linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("1D Kalman Filter Simulation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
