#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# 시뮬레이션 설정
np.random.seed(42)
n_steps = 50
true_theta = 0.5  # 실제 계수 (x * theta)

# 입력 x값과 출력 z (측정값) 생성
x_input = np.linspace(0, 10, n_steps)
noise = np.random.normal(0, 0.5, n_steps)
z_meas = true_theta * x_input + noise

# Recursive Least Squares 초기화
lambda_ = 1.0  # forgetting factor
P = 1000.0     # 공분산 초기값
theta_est = 0.0
theta_hist = []

for k in range(n_steps):
    x_k = x_input[k]
    z_k = z_meas[k]
    
    # 이득 계산
    K = P * x_k / (lambda_ + x_k * P * x_k)
    
    # 상태 업데이트
    theta_est = theta_est + K * (z_k - x_k * theta_est)
    
    # 공분산 업데이트
    P = (1 - K * x_k) * P / lambda_
    
    theta_hist.append(theta_est)

# RLS에서 예측된 z와 실제 측정 z 비교 그래프 추가
z_pred = x_input * np.array(theta_hist)

plt.figure(figsize=(12, 6))
plt.plot(z_meas, label="Measured z", alpha=0.5)
plt.plot(z_pred, label="Predicted z (RLS)", linewidth=2)
plt.plot(true_theta * x_input, label="True z (ground truth)", linestyle='--')
plt.xlabel("Time Step")
plt.ylabel("Output z")
plt.title("Measured vs Predicted Output in RLS")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# # 그래프 출력
# plt.figure(figsize=(10, 5))
# plt.plot(theta_hist, label="Estimated θ (RLS)")
# plt.hlines(true_theta, 0, n_steps, colors='r', linestyles='--', label="True θ")
# plt.xlabel("Time Step")
# plt.ylabel("Theta Estimate")
# plt.title("Recursive Least Squares Estimation of θ")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
