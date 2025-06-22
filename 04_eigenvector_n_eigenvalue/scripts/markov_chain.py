#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# 소수점 두 자리까지만 출력 설정
np.set_printoptions(precision=2, suppress=True)


P = np.array([
    [0.50, 0.20, 0.30],
    [0.30, 0.80, 0.30],
    [0.20, 0.00, 0.40]
])


print("1. set initial value")
# 초기 상태 벡터 (예시)
x0 = np.array([0.3, 0.1, 0.6])
print(f"x0: {x0}")
print("===")


print("\n2. iterative history")
# Number of iterations
num_steps = 50

# Store the evolution of the distribution
x_history = [x0]
x = x0
for _ in range(num_steps):
    x = P @ x
    x_history.append(x)

x_history = np.array(x_history)

# Compute final values to annotate
final_vals = x_history[-1]
print("===")


print("\n3. find eigen")
# Compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(P)
print(f"eigenvale: {eigvals}")
print(f"eigvectors: {eigvecs}")
print("===")

print("\n4. find weights")
V = eigvecs
V_inv = np.linalg.inv(V)
coeffs = V_inv @ x0
print(f"coeffs: {coeffs}")

sdk_coeffs = np.linalg.solve(V, x0)  # V @ c = x0 → solve for c
print(f"sdk_coeffs: {sdk_coeffs}")
print("===")

print("\n5. long term result")

# k단계 후 상태 계산: x_k = c1 * λ1^k * v1 + c2 * λ2^k * v2 + ...
def x_k(k):
    terms = [coeffs[i] * (eigvals[i]**k) * V[:, i] for i in range(3)]
    return np.sum(terms, axis=0)

# 무한대로 보낼 때 수렴값 확인
x_inf = x_k(1000)  # 충분히 큰 k
print(f"result: {x_inf}")
print("===")

print("\n6. testing")
x_inf = coeffs[0] * V[:, 0]
print(f"testing x_inf: {x_inf}")
print("===")

print("\n7. normalize eigen vector")
print("7.1. method1")
stationary_idx = np.argmin(np.abs(eigvals - 1))
stationary = eigvecs[:, stationary_idx].real
stationary /= stationary.sum()
print(f"stationary: {stationary}")
print("---")

print("7.2. method2")
eigvecs_normalized = eigvecs / eigvecs.sum(axis=0)
print(f"eigvecs_normalized: {eigvecs_normalized}")

# print(f"coeffs[0]: {coeffs[0]}")
# print(f"V[:, 0]: {V[:, 0]}")

# V[:, 1] = 0
# V[:, 2] = 0

# R = coeffs @ V
# print(f"result: {R}")


# Plot the evolution of each state
plt.figure(figsize=(10, 6))
lines = []
labels = ['State 1', 'State 2', 'State 3']
for i in range(3):
    line, = plt.plot(x_history[:, i], label=f'{labels[i]} → {final_vals[i]:.2f}')
    lines.append(line)

plt.title('Convergence to Stationary Distribution in a Markov Chain')
plt.xlabel('Step (k)')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()