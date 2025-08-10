import open3d as o3d
import cv2
import numpy as np
import laspy
from scipy.spatial.transform import Rotation as R
import time
import cupy as cp

filepath = "NUBE_PUNTOS.las"
with laspy.open(filepath) as las_file:
    las_data = las_file.read()
    pc_points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_points)

#Intrinsic parameters
focal_length = cp.array([4600, 4500])
principal_point = cp.array([975, 590])
image_size = cp.array([2000, 2000])

#Extrinsic parameters
translation = cp.array([0, 0, 3.5])
eul_angles = cp.array([259, 16.5, 80])  #Roll, pitch, yaw in degrees
rotation_matrix = R.from_euler('ZYX', eul_angles.get(), degrees=True).as_matrix()
rotation_matrix = cp.array(rotation_matrix)

video = cv2.VideoCapture("VIDEO.mp4")

combined_points = []
combined_colors = []

t = 0
inc_t = 0.5
y_inc = 0.24
y_init = 0
y_end = 20

start_time = time.time()

#Function for point cloud projection and RGB extraction with GPU acceleration
def process_points_section(section_pts, rotation_matrix, translation, focal_length, principal_point, frame):
    cam_points = (rotation_matrix @ section_pts.T).T + translation

    valid_mask = cam_points[:, 2] > 0  
    cam_points = cam_points[valid_mask]  

    x_proj = (focal_length[0] * cam_points[:, 0] / cam_points[:, 2]) + principal_point[0]
    y_proj = (focal_length[1] * cam_points[:, 1] / cam_points[:, 2]) + principal_point[1]
    x_proj = x_proj.astype(np.int32)
    y_proj = y_proj.astype(np.int32)

    proj_points = cp.column_stack((x_proj, y_proj, cam_points))

    frame_gpu = cp.array(frame)

    x_proj = proj_points[:, 0]
    y_proj = proj_points[:, 1]
    cam_points = proj_points[:, 2:]

    valid_mask = (
        (x_proj >= 0) & (x_proj < frame_gpu.shape[1]) & 
        (y_proj >= 0) & (y_proj < frame_gpu.shape[0])
    )

    valid_x = x_proj[valid_mask].astype(np.int64)
    valid_y = y_proj[valid_mask].astype(np.int64)
    valid_points = cam_points[valid_mask]

     #Extract RGB values from the frame
    rgb_values = frame_gpu[valid_y, valid_x] / 250.0  

    return valid_points, rgb_values

while y_init <= y_end:
    
    section_pts = pc_points[(pc_points[:, 1] >= y_init) & (pc_points[:, 1] <= y_init + y_inc)]

    if len(section_pts) == 0:
        continue  
    
    ret, frame = video.read()
    if not ret:
        print(f"Failed to read frame at time {t}s.")
        break
    
    cam_points, rgb_values = process_points_section(cp.array(section_pts), rotation_matrix, translation, focal_length, principal_point, frame)

    combined_points.extend(cam_points.get()) 
    combined_colors.extend(rgb_values.get())

    t += inc_t
    y_init += y_inc

#Create the new point cloud
combined_pc = o3d.geometry.PointCloud()
combined_pc.points = o3d.utility.Vector3dVector(combined_points)
combined_pc.colors = o3d.utility.Vector3dVector(combined_colors)

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(combined_pc, 0.22)

end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

o3d.io.write_point_cloud("data/combined_pc.ply", combined_pc)
print("Nube de puntos combinada exportada como combined_pc.ply")

o3d.io.write_triangle_mesh("data/mesh.ply", mesh)
print("Malla generada exportada como mesh.ply")
