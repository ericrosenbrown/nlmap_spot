import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import cv2
import open3d as o3d
from tqdm import tqdm
from spot_utils.utils import pixel_to_vision_frame

def make_pointcloud(data_path="../data/spot-depth-color-pose-data3/", pose_data_fname="pose_data.pkl", pointcloud_fname="pointcloud.pcd"):
	fill_in = False #if fill_in is True, then gaps in depth image are filled in with black (at maximal distance..?)
	save_pc = True #if true, save point cloud to same location as dir_path+dir_name

	pose_dir = pickle.load(open(f"{data_path}{pose_data_fname}","rb"))

	#######################################
	# Visualize point cloud
	file_names = os.listdir(data_path)
	num_files = int((len(file_names)-1)/ 3.0)
	total_pcds = []
	total_colors = []
	total_axes = []
	for file_num in tqdm(range(num_files)):
		rotation_matrix = pose_dir[file_num]['rotation_matrix']
		position = pose_dir[file_num]['position']

		color_img = cv2.imread(f"{data_path}color_{str(file_num)}.jpg")
		color_img = color_img[:,:,::-1]  # RGB-> BGR
		depth_img = pickle.load(open(f"{data_path}depth_{str(file_num)}","rb"))#cv2.imread(dir_path+dir_name+"depth_"+str(file_num)+".jpg")

		H,W = depth_img.shape
		for i in range(H):
			for j in range(W):
				#first apply rot2 to move camera into hand frame, then apply rotation + transform of hand frame in vision frame
				transformed_xyz,_ = pixel_to_vision_frame(i,j,depth_img,rotation_matrix,position)

				total_pcds.append(transformed_xyz)

				# Add the color of the pixel if it exists:
				if 0 <= j < W and 0 <= i < H:
					total_colors.append(color_img[i,j] / 255)
				elif fill_in:
					total_colors.append([0., 0., 0.])

		mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6,origin=[0,0,0])
		mesh_frame = mesh_frame.rotate(rotation_matrix, center=(0, 0, 0)).translate(position)
		#mesh_frame.paint_uniform_color([float(file_num)/num_files, 0.1, 1-(float(file_num)/num_files)])

		total_axes.append(mesh_frame)
		
	pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
	pcd_o3d.points = o3d.utility.Vector3dVector(total_pcds)
	pcd_o3d.colors = o3d.utility.Vector3dVector(total_colors)

	#bb = o3d.geometry.OrientedBoundingBox(center=np.array([0,0,0]),R=rot2_mat,extent=np.array([1,1,1]))

	# Visualize:
	origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=[0,0,0])
	o3d.visualization.draw_geometries([pcd_o3d]+total_axes+[origin_frame])

	if save_pc:
		o3d.io.write_point_cloud(f"{data_path}{pointcloud_fname}", pcd_o3d)

	return(pcd_o3d)
