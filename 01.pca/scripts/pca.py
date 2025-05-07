#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 포인트 클라우드 로드
pcd = o3d.io.read_point_cloud("../../data/250407/exp5_hd1080_w_marker_lift_at_corner/cam0.pcd")

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

o3d.visualization.draw_geometries([down_pcd])
