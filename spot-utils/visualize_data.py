import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import cv2

viz_poses = False

dir_name = "spot-depth-color-pose-data/"
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

intrinsics_matrix = np.array([[552.0291012161067, 0 ,320.0],[0,552.0291012161067,240.0],[0,0,1]])
file_names = os.listdir(dir_path+dir_name)
num_files = int((len(file_names)-1)/ 3.0)
for file_num in range(num_files):
	color_img = cv2.imread(dir_path+dir_name+"color_"+str(file_num)+".jpg")
	depth_img = cv2.imread(dir_path+dir_name+"depth_"+str(file_num)+".jpg")

	print(depth_img.shape)

	H,W,_ = depth_img.shape
	print(H,W)
	for h in range(H):
		for w in range(W):
			#print(w,h,depth_img[h,w])
			pass

	#Visualize color and depth

	plt.imshow(depth_img)
	plt.show()

	plt.imshow(color_img[:,:,::-1])  # RGB-> BGR
	plt.show()

	min_val = np.min(depth_img)
	max_val = np.max(depth_img)
	depth_range = max_val - min_val
	depth8 = (255.0 / depth_range * (depth_img - min_val)).astype('uint8')
	plt.imshow(depth8)
	plt.show()


