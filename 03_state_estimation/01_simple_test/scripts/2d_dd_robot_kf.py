import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent))

import math
import matplotlib.pyplot as plt
import numpy as np

from utils.plot import plot_covariance_ellipse


# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance

R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

show_animation = True

def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


##############################

def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud



def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


# import numpy as np
# import matplotlib.pyplot as plt

# # 시뮬레이션 파라미터
# dt = 0.1
# T = 20  # 총 시뮬레이션 시간 (초)
# steps = int(T / dt)

# # ground truth trajectory: 원 궤적
# radius = 5
# omega_gt = 0.2
# x_gt = np.zeros((steps, 3))  # x, y, theta

# for k in range(1, steps):
#     x_gt[k, 2] = x_gt[k-1, 2] + omega_gt * dt
#     x_gt[k, 0] = radius * np.cos(x_gt[k, 2])
#     x_gt[k, 1] = radius * np.sin(x_gt[k, 2])

# # odometry: 속도 입력 (v, w) + noise
# v_true = radius * omega_gt
# v_noisy = v_true + np.random.normal(0, 0.2, steps)
# w_noisy = omega_gt + np.random.normal(0, 0.05, steps)

# odom = np.zeros((steps, 3))
# for k in range(1, steps):
#     theta = odom[k-1, 2]
#     odom[k, 0] = odom[k-1, 0] + v_noisy[k] * np.cos(theta) * dt
#     odom[k, 1] = odom[k-1, 1] + v_noisy[k] * np.sin(theta) * dt
#     odom[k, 2] = theta + w_noisy[k] * dt

# # 센서 관측 (x, y)만 가능 + 낮은 노이즈
# sensor = x_gt[:, :2] + np.random.normal(0, 0.05, (steps, 2))

# # Kalman Filter 초기화
# x_est = np.zeros((steps, 3))
# P = np.eye(3) * 1.0

# Q = np.diag([0.5, 0.5, 0.1])**2  # odom noise
# R = np.diag([0.05, 0.05])**2     # sensor noise (신뢰도 높음)

# for k in range(1, steps):
#     # prediction (motion model)
#     theta = x_est[k-1, 2]
#     v = v_noisy[k]
#     w = w_noisy[k]
#     x_pred = x_est[k-1] + np.array([
#         v * np.cos(theta) * dt,
#         v * np.sin(theta) * dt,
#         w * dt
#     ])
#     F = np.eye(3)
#     F[0, 2] = -v * np.sin(theta) * dt
#     F[1, 2] =  v * np.cos(theta) * dt
#     P = F @ P @ F.T + Q

#     # update (sensor model: observe x, y)
#     z = sensor[k]
#     H = np.array([
#         [1, 0, 0],
#         [0, 1, 0]
#     ])
#     y = z - H @ x_pred
#     S = H @ P @ H.T + R
#     K = P @ H.T @ np.linalg.inv(S)
#     x_est[k] = x_pred + K @ y
#     P = (np.eye(3) - K @ H) @ P

# # 시각화
# plt.figure(figsize=(10, 8))
# plt.plot(x_gt[:, 0], x_gt[:, 1], label='Ground Truth Path', linewidth=2)
# plt.plot(odom[:, 0], odom[:, 1], label='Odometry Only', linestyle='--')
# plt.plot(sensor[:, 0], sensor[:, 1], '.', alpha=0.4, label='Sensor Observations')
# plt.plot(x_est[:, 0], x_est[:, 1], label='Kalman Filter Estimate', linewidth=2)
# plt.axis('equal')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Kalman Filter with Differential Drive Odometry and Noisy Position Sensor")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
