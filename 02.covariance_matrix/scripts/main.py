#!/usr/bin/env python3

import open3d as o3d # type: ignore
import numpy as np

## =====================
# PCD 파일 경로
pcd_path = "../data/point_cloud_cam0.pcd"

# PCD 파일 읽기
pcd = o3d.io.read_point_cloud(pcd_path)

# 좌표축 추가 (크기 1.0, 원점 기준)
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])


## =====================
# NumPy 배열로 변환
points = np.asarray(pcd.points)

# 모양 출력
print("Point Cloud shape:", points.shape)  # (num_points, 3)

# # 앞의 1000개 데이터 출력
# print("First 1000 points:\n", points[:1000])


## =====================
# 평균 중심화
mean = np.mean(points, axis=0)
points_centered = points - mean

print("Mean of original points (for centering):", mean)

# 공분산 행렬 계산 (표본 공분산, n-1로 나눔)
cov_matrix = np.cov(points_centered.T)  # shape: (3, 3)

print("\nCovariance matrix:")
print(cov_matrix)


## =====================
# 고유값 분해 (eigen decomposition)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors (each column is a principal axis):")
print(eigenvectors)


## =====================
# 주성분 화살표 길이 조정
arrow_scale = 0.5

# 평균점 기준으로 시각화
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=mean.tolist())

# 고유벡터를 화살표로 표시
arrows = []
for i in range(3):
    vec = eigenvectors[:, i]
    arrow = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([mean, mean + arrow_scale * vec]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    # 선 색깔 (R, G, B 순)
    color = np.zeros(3)
    color[i] = 1.0
    arrow.colors = o3d.utility.Vector3dVector([color])
    arrows.append(arrow)



## =====================
# 시각화: point cloud + 축 + 주성분 화살표
o3d.visualization.draw_geometries([pcd, axis, origin] + arrows)