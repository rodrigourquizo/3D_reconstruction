import open3d as o3d
import cv2
import numpy as np
import laspy
from scipy.spatial.transform import Rotation as R
import time
from numba import jit, prange

filepath = "NUBE_PUNTOS.las"
with laspy.open(filepath) as las_file:
    las_data = las_file.read()
    pc_points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_points)

# Intrinsic parameters
focal_length = [4600, 4500]
principal_point = [975, 590]
image_size = [2000, 2000]

# Extrinsic parameters
translation = np.array([0, 0, 3.5])
eul_angles = [259, 16.5, 80]  # Roll, pitch, yaw in degrees
rotation_matrix = R.from_euler('ZYX', eul_angles, degrees=True).as_matrix()

video = cv2.VideoCapture("VIDEO.mp4")

combined_points = []
combined_colors = []

t = 0
inc_t = 0.5
y_inc = 0.24
y_init = 0
y_end = 20

start_time = time.time()

while y_init <= y_end:
    #Filter the points on the y-axis
    section_pts = pc_points[(pc_points[:, 1] >= y_init) & (pc_points[:, 1] <= y_init + y_inc)]
    
    if len(section_pts) == 0:
        continue  # Skip empty sections

    #Point cloud projection
    cam_points = (rotation_matrix @ section_pts.T).T + translation

    #Filter points behind the camera
    valid_mask = cam_points[:, 2] > 0  # Mask for points with Z > 0
    cam_points = cam_points[valid_mask]  # Filter valid points

    #Project to 2D image space
    x_proj = (focal_length[0] * cam_points[:, 0] / cam_points[:, 2]) + principal_point[0]
    y_proj = (focal_length[1] * cam_points[:, 1] / cam_points[:, 2]) + principal_point[1]
    x_proj = x_proj.astype(np.int32)
    y_proj = y_proj.astype(np.int32)

    #Create array for projected points (X, Y, Z)
    proj_points = np.column_stack((x_proj, y_proj, cam_points))

    video.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = video.read()
    if not ret:
        print(f"Failed to read frame at time {t}s.")
        break

    #Extract RGB values from the frame
    x_proj = proj_points[:, 0]
    y_proj = proj_points[:, 1]
    cam_points = proj_points[:, 2:]

    valid_mask = (
        (x_proj >= 0) & (x_proj < frame.shape[1]) & 
        (y_proj >= 0) & (y_proj < frame.shape[0])
    )

    valid_x = x_proj[valid_mask].astype(np.int64)
    valid_y = y_proj[valid_mask].astype(np.int64)
    valid_points = cam_points[valid_mask]

    rgb_values = frame[valid_y, valid_x] / 255.0  # Normalize RGB

    combined_points.extend(valid_points)
    combined_colors.extend(rgb_values)

    t += inc_t
    y_init += y_inc

#Create the new point cloud
combined_pc = o3d.geometry.PointCloud()
combined_pc.points = o3d.utility.Vector3dVector(combined_points)
combined_pc.colors = o3d.utility.Vector3dVector(combined_colors)

# Poisson surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(combined_pc, 0.22)

end_time = time.time()
print(f"Tiempo de ejecucion: {end_time - start_time:.2f} seconds")

#o3d.visualization.draw_geometries([mesh])

"""o3d.io.write_point_cloud("data/combined_pc.ply", combined_pc)
print("Nube de puntos combinada exportada como combined_pc.ply")

o3d.io.write_triangle_mesh("data/mesh.ply", mesh)
print("Malla generada exportada como mesh.ply")"""

