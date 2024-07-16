#%% 1. Env setup

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt



#%% 2 data prep

Dataname = ("appartment_cloud.ply")
pcd = o3d.io.read_point_cloud("./tutorial_data/" + Dataname)



#%% 3 pre-processing
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

#%% 3.1 statistical outlier filter
nn = 16
std_multiplier = 10

filtered_pcd = pcd.remove_statistical_outlier(nn, std_multiplier)

outliers = pcd.select_by_index(filtered_pcd[1], invert=True)

outliers.paint_uniform_color([1, 0, 0])
filtered_pcd = filtered_pcd[0]

o3d.visualization.draw_geometries([filtered_pcd])

#%% 3.2 voxel downsamplin
voxel_size = 0.01

pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)
o3d.visualization.draw_geometries([pcd_downsampled])

#%% 3.3 estimating normals
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())

radius_normals = nn_distance * 4

pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn = 16), fast_normal_computation=True)

pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled, outliers])

#%% 4 extracting and setting parameters

front = 0.98924968999074026, 0.11976321498654677, 0.083915571792787427
lookat = -0.0087864400828863154, 0.02734387141551986, -0.03272785576983317
up = -0.085617819753286639, 0.0091302998492761977, 0.99628621719130295
zoom = 0.27999999999999958
pcd = pcd.voxel_down_sample(voxel_size = voxel_size)
o3d.visualization.draw_geometries([pcd], front=front, up=up, zoom=zoom,lookat=lookat)

#%% 5 ransac planar segmentation first ver

pt_to_plane_dist = 0.02

plane_model, inliers = pcd.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=1000)

[a,b,c,d] = plane_model

print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outliers_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outliers_cloud.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries([inlier_cloud, outliers_cloud], front=front, up=up, zoom=zoom,lookat=lookat)

#%% 5.2 ransac planar segmentation 2nd ver

pt_to_plane_dist = 0.02

plane_model, inliers = pcd.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=1000)

[a,b,c,d] = plane_model

print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outliers_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outliers_cloud.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries([inlier_cloud, outliers_cloud], front=front, up=up, zoom=zoom,lookat=lookat)

#%% 6 multi-order ransac
max_plane_idx = 6
pt_to_plane_dist = 0.02

segment_models = {}
segments = {}
rest = pcd
for i in range(max_plane_idx):
    colors = plt.get_cmap('tab20')(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=1000)
    segments[i] = rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass", i, "/", max_plane_idx, "done. ")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest], zoom=zoom, front=front, up=up, lookat=lookat)


#%% DBScan sur rest

labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest], zoom=zoom, front=front, up=up, lookat=lookat)