#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

print("1. Load training images (20 images of 45x40)")
print("")
# 1. Load training images (20 images of 45x40)
n = 20
h, w = 45, 40
x = np.zeros((n, h * w))

print(f"x type: {type(x)}")
print(f"x shape: {x.shape}")

for i in range(n):
    img = Image.open(f"../data/{i+1}.png").convert('L')  # Grayscale
    x[i, :] = np.asarray(img).reshape(-1)

# 1번 이미지 시각화 (reshape to 45x40)
plt.imshow(x[0].reshape(h, w), cmap='gray')
plt.title("Image 1")
plt.axis('off')
plt.show()

print("")
print("2. Subtract mean face")
print("")

# 2. Subtract mean face
mean_face = np.mean(x, axis=0)
print(f"mean_face type: {type(mean_face)}")
print(f"mean_face shape: {mean_face.shape}")
x_centered = x - mean_face
print("")
print(f"x_centered type: {type(x_centered)}")
print(f"x_centered shape: {x_centered.shape}")

# 원본 이미지와 중심화 이미지 비교 (concatenation)
original = x[0].reshape(h, w)
centered = x_centered[0].reshape(h, w)

# 중심화 이미지의 값을 시각화에 적합하도록 정규화 (선택적, 보기 좋게)
centered_norm = (centered - centered.min()) / (centered.max() - centered.min()) * 255

# 좌우로 붙이기
comparison = np.concatenate([original, centered_norm], axis=1)

# 시각화
plt.imshow(comparison, cmap='gray')
plt.title("Original (Left) vs Centerized (Right)")
plt.axis('off')
plt.show()

print("")
print("3. Compute covariance and eigen decomposition")
print("")

# 3. Compute covariance and eigen decomposition
cov_mat = np.cov(x_centered, rowvar=False)
print(f"cov_mat type: {type(cov_mat)}")
print(f"cov_mat shape: {cov_mat.shape}")
eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
print("")
print(f"eig_vals type: {type(eig_vals)}")
print(f"eig_vals shape: {eig_vals.shape}")
print("")
print(f"eig_vecs type: {type(eig_vecs)}")
print(f"eig_vecs shape: {eig_vecs.shape}")

print("")
print("4. Sort eigenvectors by descending eigenvalues")
print("")

# 4. Sort eigenvectors by descending eigenvalues
sorted_idx = np.argsort(eig_vals)[::-1]
print(f"sorted idx: {sorted_idx}")

eig_vals = eig_vals[sorted_idx]
eig_vecs = eig_vecs[:, sorted_idx]

print("")
print("5. Reconstruction with top-k eigenfaces")
print("")

# 5. Reconstruction with top-k eigenfaces
for k in [20, 10, 5, 2]:
    print(f"k: {k}")
    eig_k = eig_vecs[:, :k]
    print(f"eig_k type: {type(eig_k)}")
    print(f"eig_k shape: {eig_k.shape}")
    y_k = x_centered @ eig_k         # Projected coordinates
    x_recons = y_k @ eig_k.T         # Reconstruct
    x_recons += mean_face            # Add mean back

    # Save or show reconstructed images
    os.makedirs(f"recons_k{k}", exist_ok=True)
    for i in range(n):
        img = x_recons[i].reshape(h, w)
        img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(f"recons_k{k}/{i+1}_recon.png")
    print("")