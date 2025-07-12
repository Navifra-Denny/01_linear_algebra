#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 간단한 2x2 행렬 정의
A = np.array([[2, 3],
              [1, 4],
              [3, 2]])

# 수식 기반으로 QR 분해 수행 (Gram-Schmidt)
a1 = A[:, 0]
a2 = A[:, 1]
a3 = A[:, 2]

print(f"A: {A}")
print(f"a1: {a1}")
print(f"a2: {a2}")
print(f"a3: {a3}")

# q1 = a1 / ||a1||
q1 = a1 / np.linalg.norm(a1)
print("\n=====\n")
print(f"q1 = a1 / ||a1||: {q1}")

# hat{a2} = proj_q1_a2 = (q1^T * a2) * q1
proj_q1_a2 = np.dot(q1, a2) * q1
print("\n=====\n")
print(f"proj_q1_a2 = (q1^T * a2) * q1: {proj_q1_a2}")

# a2-hat{a2}
u2 = a2 - proj_q1_a2
q2 = u2 / np.linalg.norm(u2)
print("\n=====\n")
print(f"q2 = u2 / ||u2||: {q2}")

Q = np.column_stack((q1, q2))
print("\n====\n")
print(f"Q: {Q}")

R = Q.T @ A
print("\n====\n")
print(f"R: {R}")


# R[0,0]*q1 = A[:,0] 방향 벡터
a1_recon = R[0,0] * q1

# R[0,1]*q1 + R[1,1]*q2 = A[:,1] 방향 벡터
a2_recon = R[0,1] * q1 + R[1,1] * q2

# A 복원 벡터들
A_recon_vectors = [a1_recon, a2_recon]


# # QR 분해 수행
# Q, R = np.linalg.qr(A)

# # 결과 출력
# print("Matrix A:")
# print(A)

# print("\nOrthogonal Matrix Q:")
# print(Q)

# print("\nUpper Triangular Matrix R:")
# print(R)

# 시각화를 위한 함수
def plot_vectors(vectors, colors, title):
    fig, ax = plt.subplots()
    origin = np.zeros(2)
    for vec, color in zip(vectors, colors):
        ax.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1, color=color)
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title)
    plt.show()

# # 시각화: A와 Q의 열 벡터
# plot_vectors([A[:,0], A[:,1]], ['red', 'orange'], 'Original Matrix A Columns')
# plot_vectors([Q[:,0], Q[:,1]], ['blue', 'green'], 'Orthonormal Matrix Q Columns')

# # 시각화를 위한 함수 (A와 Q를 동시에 표시)
# def plot_vectors_combined(vectors1, vectors2, colors1, colors2, labels1, labels2, title):
#     fig, ax = plt.subplots()
#     origin = np.zeros(2)
    
#     for vec, color, label in zip(vectors1, colors1, labels1):
#         ax.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1, color=color, label=label)
    
#     for vec, color, label in zip(vectors2, colors2, labels2):
#         ax.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1, color=color, label=label, linestyle='dashed')
    
#     ax.set_xlim(-1, 3)
#     ax.set_ylim(-1, 3)
#     ax.set_aspect('equal')
#     ax.grid(True)
#     ax.set_title(title)
#     ax.legend()
#     plt.show()

# 벡터 정보
A_vectors = [A[:, 0], A[:, 1]]
Q_vectors = [Q[:, 0], Q[:, 1]]
A_colors = ['red', 'orange']
Q_colors = ['blue', 'green']
A_labels = ['A col1', 'A col2']
Q_labels = ['Q col1', 'Q col2']
# 색상 및 라벨
R_colors = ['purple', 'brown']
R_labels = ['R[0,0]*q1', 'R[:,1]*Q']


# 시각화
def plot_with_reconstruction():
    fig, ax = plt.subplots()
    origin = np.zeros(2)

    # 원래 A 벡터
    for vec, color, label in zip(A_vectors, A_colors, A_labels):
        ax.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1, color=color, label=label)

    # Q 벡터
    for vec, color, label in zip(Q_vectors, Q_colors, Q_labels):
        ax.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1, color=color, linestyle='dotted', label=label)

    # R*Q 조합 벡터 (A 복원)
    for vec, color, label in zip(A_recon_vectors, R_colors, R_labels):
        ax.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1, color=color, linestyle='dashdot', label=label)

    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('A vs Q Columns vs R-weighted Q (A reconstruction)')
    ax.legend()
    plt.show()


# 한 그림에 함께 표시
plot_with_reconstruction()


# plot_vectors_combined(A_vectors, Q_vectors, A_colors, Q_colors, A_labels, Q_labels, 'A vs. Q Columns (QR Decomposition)')
