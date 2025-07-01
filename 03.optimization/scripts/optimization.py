#!/usr/bin/env python3

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from open3d.pipelines.registration import PoseGraph, PoseGraphEdge, PoseGraphNode

print("##############################")
print(f"1. --- get initial pose ---")
pose_params_deg = [
    [0.818, -2.261, 4.348, 154.923, 8.649, -96.185],
    [2.827, -2.382, 4.306, -179.594, -14.738, 92.306],
    [4.808, -2.426, 4.476, 158.203, -10.613, 88.533],
    [5.024, -6.394, 4.267, 161.058, 3.785, 90.528],
    [2.836, -6.350, 4.318, -179.167, 10.439, 86.588],
    [0.738, -6.140, 4.316, 158.110, -5.158, -82.654]
]

def make_transform(x, y, z, roll_deg, pitch_deg, yaw_deg):
    # 1. 회전: RPY (deg) → radians
    r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
    rot_matrix = r.as_matrix()

    # 2. 변환 행렬 구성
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = [x, y, z]
    return T

initial_poses = [make_transform(*pose) for pose in pose_params_deg]
print(initial_poses)

def preprocess_point_cloud(pcd, voxel_size=0.05):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd_down


def pairwise_registration(source, target, init=np.identity(4)):
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=0.1,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return icp_result.transformation, icp_result.fitness


print("\n##############################")
print("2. --- visualize only initial poses ---")

initial_map = o3d.geometry.PointCloud()

for i in range(6):
    pcd = o3d.io.read_point_cloud(f"../../data/250407/exp6_hd1080_w_marker_lift_at_center/cam{i}.pcd")
    pcd.transform(initial_poses[i])  # 초기 pose 그대로 적용
    initial_map += pcd

o3d.visualization.draw_geometries([initial_map])

# print("\n##############################")
# print("2. --- get pose graph ---")

# pose_graph = PoseGraph()
# pose_graph.nodes.append(PoseGraphNode(np.identity(4)))  # 첫 노드 고정

# for i in range(5):
#     source = o3d.io.read_point_cloud(f"../../data/250407/exp6_hd1080_w_marker_lift_at_center/cam{i}.pcd")
#     target = o3d.io.read_point_cloud(f"../../data/250407/exp6_hd1080_w_marker_lift_at_center/cam{i+1}.pcd")

#     source = preprocess_point_cloud(source)
#     target = preprocess_point_cloud(target)

#     init_guess = np.linalg.inv(initial_poses[i]) @ initial_poses[i+1]
#     trans, fitness = pairwise_registration(source, target, init=init_guess)

#     pose_graph.nodes.append(PoseGraphNode(np.linalg.inv(trans)))
#     pose_graph.edges.append(PoseGraphEdge(i, i+1, trans, uncertain=False))

# print(pose_graph)
# print("\n2.1. --- PoseGraph Nodes (Absolute poses) ---")
# for idx, node in enumerate(pose_graph.nodes):
#     print(f"Node {idx}:")
#     print(node.pose)


# # print("\n2.2. --- PoseGraph Edges (Relative transforms) ---")
# # for idx, edge in enumerate(pose_graph.edges):
# #     print(f"Edge {idx}: from node {edge.source_node} to node {edge.target_node}")
# #     print("Transformation:")
# #     print(edge.transformation)
# #     print(f"Uncertain: {edge.uncertain}")


# o3d.pipelines.registration.global_optimization(
#     pose_graph,
#     o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
#     o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
#     o3d.pipelines.registration.GlobalOptimizationOption(
#         max_correspondence_distance=0.1,
#         edge_prune_threshold=0.25,
#         reference_node=0
#     )
# )


# final_map = o3d.geometry.PointCloud()

# for i in range(6):
#     pcd = o3d.io.read_point_cloud(f"../../data/250407/exp6_hd1080_w_marker_lift_at_center/cam{i}.pcd")
#     pcd.transform(pose_graph.nodes[i].pose)
#     final_map += pcd

# o3d.visualization.draw_geometries([final_map])
