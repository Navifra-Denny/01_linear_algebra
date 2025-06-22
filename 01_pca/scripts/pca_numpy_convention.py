#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


print("----------------------------------------------------------")
print("1. read pcd & visualization as downsampled with clustering")

# 포인트 클라우드 로드
pcd = o3d.io.read_point_cloud("../../data/250407/cam0.bottom.pcd")
# pcd = o3d.io.read_point_cloud("/home/yeongsoo/workspaces/infra_ws/reconstruction/data/250407/cam0.bottom.pcd")
o3d.visualization.draw_geometries([pcd])

# Voxel downsampling (voxel 크기 지정 가능)
voxel_size = 0.015  # 예: 2cm 크기의 voxel
down_pcd = pcd.voxel_down_sample(voxel_size)

# 포인트 클라우드에서 노멀 벡터 계산 (필수)
down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))

# Region Growing Segmentation을 이용한 군집화
labels = np.array(down_pcd.cluster_dbscan(eps=0.03, min_points=5, print_progress=True))

# 결과 시각화 (군집 색상으로 시각화)
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # 노이즈는 검정색 처리
down_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 좌표축 프레임 생성 (원점에 길이 0.1의 축 표시)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# 포인트 클라우드와 함께 시각화
o3d.visualization.draw_geometries([down_pcd, coordinate_frame])

print("")
print("----------------------------------------------------------")
print("2. generate mean-deviation from")

points = np.asarray(down_pcd.points)
print(f"type of points: {type(points)}")
print(f"shape of points: {points.shape}")
print("# data form n x d (n: number of data, d: feature dimension)")

# 평균 계산
mean = points.mean(axis=0)

# 중심화 (mean-deviation form)
centered_points = points - mean

print("")
print("Point cloud 평균:", mean)
print("Centered point cloud shape:", centered_points.shape)

# 중심화된 포인트 클라우드 생성
centered_pcd = o3d.geometry.PointCloud()
centered_pcd.points = o3d.utility.Vector3dVector(centered_points)

# 색상 지정 (회색)
gray = np.tile([0.5, 0.5, 0.5], (centered_points.shape[0], 1))
centered_pcd.colors = o3d.utility.Vector3dVector(gray)

# 시각화
o3d.visualization.draw_geometries([down_pcd, centered_pcd, coordinate_frame])


print("")
print("----------------------------------------------------------")
print("3. covariance matrix calculated by B^T B")

# centered_points를 B로 명명
B = centered_points  # B: (n × d) matrix

print("3.1 calculating the covariance without library. covariance matrix calculated by B^T B")

# 공분산 행렬 계산: (1 / (n - 1)) * B^T B
n_samples = B.shape[0]
cov_matrix = (B.T @ B) / (n_samples - 1)

print(f"B shape: {B.shape}")
print("Covariance matrix (3x3):\n", cov_matrix)

print("")
print("3.2 calculating the covariance with library np.cov.")
# 공분산 행렬 계산 (X: n × d → 공분산: d × d)
cov_matrix_w_lib = np.cov(centered_points, rowvar=False)
print("Covariance matrix (3x3):\n", cov_matrix_w_lib)


print("")
print("----------------------------------------------------------")
print("4. Calculating eigenvector, eigenvalue")

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 고유값 내림차순 정렬
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"eigen values shape:\n", eigenvalues.shape)
print(f"eigen vectors shape:\n", eigenvectors.shape)
print("eigen values:\n", eigenvalues)
print("eigen vectors:\n", eigenvectors)

# PCA 방향 벡터 시각화용 길이 스케일
scale = 0.5 * np.max(eigenvalues)

# 각 주성분 벡터를 원점에서 시작하는 선분으로
lines = []
colors = []
points_for_arrow = []

for i in range(3):
    start = [0., 0., 0.]
    end = [0., 0., 0.] + eigenvectors[:, i] * eigenvalues[i] * scale
    points_for_arrow.append(start)
    points_for_arrow.append(end)
    lines.append([2*i, 2*i+1])
    colors.append([1, 0, 0])  # 빨간색 (3개 모두 동일 색상 사용)

# LineSet으로 화살표 형태 시각화
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points_for_arrow)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([down_pcd, centered_pcd, coordinate_frame, line_set])


print("")
print("----------------------------------------------------------")
print("5. Visualizing PCA-transformed data")

print("5.1. 모든 주성분 사용")
# 모든 주성분 사용
Z = B @ eigenvectors  # 회전된 좌표계 (n × 3)

# PCA 좌표계에 투영된 포인트 클라우드 생성
pca_aligned_pcd = o3d.geometry.PointCloud()
pca_aligned_pcd.points = o3d.utility.Vector3dVector(Z)

# 색상 지정 (녹색)
green = np.tile([0.0, 1.0, 0.0], (Z.shape[0], 1))
pca_aligned_pcd.colors = o3d.utility.Vector3dVector(green)

o3d.visualization.draw_geometries([down_pcd, centered_pcd, pca_aligned_pcd, coordinate_frame, line_set])


print("5.2. 상위 2개 주성분만 선택 (d=3 → k=2)")
# 상위 2개 주성분만 선택 (d=3 → k=2)
V2 = eigenvectors[:, :2]  # shape: (3, 2)

# 2D PCA 좌표계로 투영
Z2D = B @ V2  # shape: (n, 2)

# 2D → 3D로 확장 (z축에 0 추가)
Z2D_3D = np.hstack((Z2D, np.zeros((Z2D.shape[0], 1))))  # shape: (n, 3)

# 포인트 클라우드 생성
pca_2D_pcd = o3d.geometry.PointCloud()
pca_2D_pcd.points = o3d.utility.Vector3dVector(Z2D_3D)

# 색상 지정 (파란색)
blue = np.tile([0.0, 0.0, 1.0], (Z2D.shape[0], 1))
pca_2D_pcd.colors = o3d.utility.Vector3dVector(blue)


o3d.visualization.draw_geometries([
    down_pcd,
    centered_pcd,
    pca_aligned_pcd,    # 3D PCA 전체 투영 (녹색)
    pca_2D_pcd,         # 상위 2D PCA 투영 (파란색)
    coordinate_frame,
    line_set
])



print("")
print("----------------------------------------------------------")
print("6. Visualizing reconstruction data")

# 1. 역회전: Z2D (n×2) → B 복원 (n×3)
B_hat = Z2D @ V2.T  # shape: (n, 3)

# 2. 평균 복원
X_hat = B_hat + mean  # shape: (n, 3)

# 복원된 포인트 클라우드 생성
reconstructed_pcd = o3d.geometry.PointCloud()
reconstructed_pcd.points = o3d.utility.Vector3dVector(X_hat)

# 색상 지정 (보라색)
purple = np.tile([0.6, 0.0, 0.6], (X_hat.shape[0], 1))
reconstructed_pcd.colors = o3d.utility.Vector3dVector(purple)

o3d.visualization.draw_geometries([
    down_pcd,
    centered_pcd,
    pca_aligned_pcd,
    pca_2D_pcd,
    reconstructed_pcd,   # 복원된 포인트 클라우드
    coordinate_frame,
    line_set
])