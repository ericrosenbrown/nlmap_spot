import numpy as np

def pixel_to_vision_frame_depth_provided(i,j,depth,rotation_matrix,position):
	'''
	Converts a pixel (i,j) in HxW image to 3d position in vision frame

	i,j: pixel location in image
	depth_img: HxW depth image
	rotaton_matrix: 3x3 rotation matrix of hand in vision frame
	position: 3x1 position vector of hand in vision frame
	'''

	#hand_tform_camera comes from line below, just a hardcoded version of it
	#rot2 = mesh_frame.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))
	hand_tform_camera = np.array([[ 3.74939946e-33,6.12323400e-17,1.00000000e+00],
	[-1.00000000e+00,6.12323400e-17,0.00000000e+00],
	[-6.12323400e-17,-1.00000000e+00,6.12323400e-17]])

	#Intrinsics for RGB hand camera on spot
	CX = 320
	CY = 240
	FX= 552.0291012161067
	FY = 552.0291012161067


	z_RGB = depth
	x_RGB = (j - CX) * z_RGB / FX
	y_RGB = (i - CY) * z_RGB / FY

	bad_z = z_RGB == 0 #if z_RGB is 0, the depth was 0, which means we didn't get a real point. x,y,z will just be where robot hand was

	#first apply rot2 to move camera into hand frame, then apply rotation + transform of hand frame in vision frame
	transformed_xyz = np.matmul(rotation_matrix,np.matmul(hand_tform_camera,np.array([x_RGB,y_RGB,z_RGB]))) + position

	return(transformed_xyz,bad_z)


def pixel_to_vision_frame(i,j,depth_img,rotation_matrix,position):
	'''
	Converts a pixel (i,j) in HxW image to 3d position in vision frame

	i,j: pixel location in image
	depth_img: HxW depth image
	rotaton_matrix: 3x3 rotation matrix of hand in vision frame
	position: 3x1 position vector of hand in vision frame
	'''

	#hand_tform_camera comes from line below, just a hardcoded version of it
	#rot2 = mesh_frame.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))
	hand_tform_camera = np.array([[ 3.74939946e-33,6.12323400e-17,1.00000000e+00],
	[-1.00000000e+00,6.12323400e-17,0.00000000e+00],
	[-6.12323400e-17,-1.00000000e+00,6.12323400e-17]])

	#Intrinsics for RGB hand camera on spot
	CX = 320
	CY = 240
	FX= 552.0291012161067
	FY = 552.0291012161067


	z_RGB = depth_img[i,j]
	x_RGB = (j - CX) * z_RGB / FX
	y_RGB = (i - CY) * z_RGB / FY

	bad_z = z_RGB == 0 #if z_RGB is 0, the depth was 0, which means we didn't get a real point. x,y,z will just be where robot hand was

	#first apply rot2 to move camera into hand frame, then apply rotation + transform of hand frame in vision frame
	transformed_xyz = np.matmul(rotation_matrix,np.matmul(hand_tform_camera,np.array([x_RGB,y_RGB,z_RGB]))) + position

	return(transformed_xyz,bad_z)
