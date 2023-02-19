import pickle
import matplotlib.pyplot as plt
import math

dir_name = "spot-depth-color-pose-data/"
dir_path = "../"

pose_data_fname = "pose_data.pkl"
pose_dir = pickle.load(open(dir_path+dir_name+pose_data_fname,"rb"))

#######################################
# Graph robot poses
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

