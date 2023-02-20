import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import cv2
import open3d as o3d

viz_poses = True

dir_name = "spot-depth-color-pose-data2/"
dir_path = "../data/"

pose_data_fname = "pose_data.pkl"
pose_dir = pickle.load(open(dir_path+dir_name+pose_data_fname,"rb"))

#######################################
# Graph robot poses
if viz_poses:
	x_positions = []
	y_positions = []
	x_rots = []
	y_rots = []
	labels = []
	for (counter,pose) in pose_dir.items():
		x_pos = pose['position'][0]
		y_pos = pose['position'][1]
		yaw = pose['rpy'][2] #euler angles in radians
		x_rot = math.cos(yaw)
		y_rot = math.sin(yaw)


		labels.append(counter)
		x_positions.append(x_pos)
		y_positions.append(y_pos)
		x_rots.append(x_rot)
		y_rots.append(y_rot)

	plt.scatter(x_positions,y_positions)
	plt.quiver(x_positions,y_positions,x_rots,y_rots)
	for i, label in enumerate(labels):
		plt.annotate(label, (x_positions[i], y_positions[i]))
	plt.show()

#######################################
# Visualize point cloud
CX = 320
CY = 240
FX= 552.0291012161067
FY = 552.0291012161067

#rot2_mat comes from rot2 line below, just a hardcoded version of it, it's the rotation from camera to hand frame
#rot2 = mesh_frame.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))
rot2_mat = np.array([[ 3.74939946e-33,6.12323400e-17,1.00000000e+00],
 [-1.00000000e+00,6.12323400e-17,0.00000000e+00],
 [-6.12323400e-17,-1.00000000e+00,6.12323400e-17]])


R = np.array([[FX, 0 ,CX],[0,FY,CY],[0,0,1]])
T = np.array([1,1,2])
file_names = os.listdir(dir_path+dir_name)
num_files = int((len(file_names)-1)/ 3.0)
total_pcds = []
total_axes = []
for file_num in range(num_files):
	total_colors = []

	color_img = cv2.imread(dir_path+dir_name+"color_"+str(file_num)+".jpg")
	color_img = color_img[:,:,::-1]  # RGB-> BGR
	depth_img = pickle.load(open(dir_path+dir_name+"depth_"+str(file_num),"rb"))#cv2.imread(dir_path+dir_name+"depth_"+str(file_num)+".jpg")

	print(depth_img.shape)

	H,W = depth_img.shape
	print(H,W)
	pcd = []
	colors = []
	for i in range(H):
		for j in range(W):
			
			z_RGB = depth_img[i,j]
			x_RGB = (j - CX) * z_RGB / FX
			y_RGB = (i - CY) * z_RGB / FY

			#first apply rot2 to move camera into hand frame, then apply rotation + transform of hand frame in vision frame
			transformed_xyz = np.matmul(pose_dir[file_num]['rotation_matrix'],np.matmul(rot2_mat,np.array([x_RGB,y_RGB,z_RGB]))) + pose_dir[file_num]['position']

			#print(i,j,depth_img[i,j],z_RGB)

			"""
			Convert from rgb camera coordinates system
			to rgb image coordinates system:
			"""

			if z_RGB != 0:
				j_rgb = int((x_RGB * FX) / z_RGB + CX)
				i_rgb = int((y_RGB * FY) / z_RGB + CY)

				# Add point to point cloud:
				#pcd.append([x_RGB, y_RGB, z_RGB])
				pcd.append(transformed_xyz)

				# Add the color of the pixel if it exists:
				if 0 <= j_rgb < W and 0 <= i_rgb < H:
					colors.append(color_img[i_rgb,j_rgb] / 255)
				else:
					colors.append([0., 0., 0.])

	mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6,origin=[0,0,0])
	mesh_frame = mesh_frame.rotate(rot2_mat,center=(0,0,0)).rotate(pose_dir[file_num]['rotation_matrix'], center=(0, 0, 0)).translate(pose_dir[file_num]['position'])
	#mesh_frame.paint_uniform_color([float(file_num)/num_files, 0.1, 1-(float(file_num)/num_files)])

	total_axes.append(mesh_frame)
	
	pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
	pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
	pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

	total_pcds.append(pcd_o3d)
	# Convert to Open3D.PointCLoud
	'''
	pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
	pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
	pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
	# Visualize:
	o3d.visualization.draw_geometries([pcd_o3d,mesh_frame])
	'''
	#Visualize color and depth

	'''
	plt.imshow(depth_img)
	plt.show()

	plt.imshow(color_img)
	plt.show()

	min_val = np.min(depth_img)
	max_val = np.max(depth_img)
	depth_range = max_val - min_val
	depth8 = (255.0 / depth_range * (depth_img - min_val)).astype('uint8')
	plt.imshow(depth8)
	plt.show()
	'''

# Convert to Open3D.PointCLoud:
#pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
#pcd_o3d.points = o3d.utility.Vector3dVector(total_pcd)
#pcd_o3d.colors = o3d.utility.Vector3dVector(total_colors)
# Visualize:
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=[0,0,0]).rotate(rot2_mat,center=(0,0,0))
o3d.visualization.draw_geometries(total_pcds+total_axes+[origin_frame])