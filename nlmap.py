from vild.vild_utils import *
import pickle
import os
from collections import defaultdict
import math
from queue import PriorityQueue
from tqdm import tqdm
import open3d as o3d
from spot_utils.utils import pixel_to_vision_frame, pixel_to_vision_frame_depth_provided, arm_object_grasp
from nlmap_utils import get_best_clip_vild_dirs
from spot_utils.move_spot_to import move_to 

from bosdyn.client.image import ImageClient

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import geometry_pb2
from bosdyn.api import basic_command_pb2

from bosdyn.client.frame_helpers import VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                        block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import math_helpers
from bosdyn.client.manipulation_api_client import ManipulationApiClient



#################################################################
# Hyperparameters and general initialization
cache_images = True #load image cache if available, make image cache when needed
cache_text = True#same but for text
vis_boxes = False #show image with detected bounding boxes
vis_details = False #show details for each bounding box
headless = True #no visualization at all
top_k = 5 #top k scores for models get stored
img_dir_root_path = "./data/"
img_dir_name = "spot-depth-color-pose-data8"
img_dir_path = img_dir_root_path + img_dir_name
cache_path = "./cache/"
pose_data_fname = "pose_data.pkl"
hostname = "138.16.161.12" #TODO pass this in

numbered_categories = [{'name': str(idx), 'id': idx,} for idx in range(50)]
numbered_category_indices = {cat['id']: cat for cat in numbered_categories}

#################################################################
# Assuming point cloud already exists at location, load it in along with pose data
pcd = o3d.io.read_point_cloud(img_dir_root_path+img_dir_name+"/pointcloud.pcd")
pose_dir = pickle.load(open(img_dir_root_path+img_dir_name+"/"+pose_data_fname,"rb"))
#o3d.visualization.draw_geometries([pcd])
#################################################################
# Preprocessing categories and get params
'''
category_name_string = ';'.join(['flipflop', 'street sign', 'bracelet',
    'necklace', 'shorts', 'floral camisole', 'orange shirt',
    'purple dress', 'yellow tee', 'green umbrella', 'pink striped umbrella', 
    'transparent umbrella', 'plain pink umbrella', 'blue patterned umbrella',
    'koala', 'electric box','car', 'pole'])
'''
#category_name_string = "Table; Chair; Sofa; Lamp; Rug; Television; Fireplace; Pillow; Blanket; Clock; Picture frame; Vase; Lampshade; Candlestick; Books; Magazines; DVD player; CD player; Record player; Video game console; Board game; Card game; Chess set; Backgammon set; Carpet; Drapes; Blinds; Shelving unit; Side table; Coffee table; Footstool; Armchair; Bean bag; Desk; Office chair; Computer; Printer; Scanner; Fax machine; Telephone; Cell phone; Rug; Trash can; Wastebasket; Vacuum cleaner; Broom; Dustpan; Mop; Bucket; Dust cloth; Cleaning supplies; Iron; Ironing board; Hair dryer; Curling iron; Toilet brush; Towels; Soap; Shampoo; Toothbrush; Toothpaste; Razor; Shaving cream; Deodorant; Hairbrush; Hair ties; Makeup; Nail polish; Perfume; Cologne; Laundry basket; Clothes hanger; Closet; Dresser; Bed; Mattress; Pillows; Sheets; Blanket; Comforter; Quilt; Bedspread; Nightstand; Alarm clock; Lamp; Lamp; Rug"
#category_name_string = "Hairbrush; Lamp; Chair; Sofa; Books; Television" 
#category_name_string = "Boxes; Books; Monitor; Lamp" 
category_name_string = "Expo marker bottle; cup"#; Chair; Boxes; Door; Table; Picture; Something to sit on " 

#category_name_string = "food; chair; person; sofa; pillow; table; book"

category_names = [x.strip() for x in category_name_string.split(';')]
#category_names = ['background'] + category_names

fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)

#################################################################
# Loop through images at path locations
img_names = os.listdir(img_dir_path)
img_names = [img_name for img_name in img_names if "color" in img_name]


priority_queue_vild_dir, priority_queue_clip_dir = get_best_clip_vild_dirs(img_names,img_dir_path,cache_images=cache_images,cache_text=cache_text,cache_path=cache_path,img_dir_name=img_dir_name,category_names=category_names,headless=headless)


#################################################################
# Spot setup
sdk = bosdyn.client.create_standard_sdk('NLMapSpot')
robot = sdk.create_robot(hostname)
bosdyn.client.util.authenticate(robot)

# Time sync is necessary so that time-based filter requests can be converted
robot.time_sync.wait_for_sync()

assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                "such as the estop SDK example, to configure E-Stop."

robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
image_client = robot.ensure_client(ImageClient.default_service_name)
manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

#We only use the hand color
sources = ['hand_depth_in_hand_color_frame', 'hand_color_image']

#################################################################
# Loop through objects

with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
	#print(category_names_vildclip_dir)
	for category_name in category_names:
		print(f"category: {category_name}")
		top_axes = []
		if not headless:
			fig, axs = plt.subplots(2, top_k)
			plt.suptitle(f"Query: {category_name}")
		best_pose = None
		for k in range(top_k):
			top_k_item_vild = priority_queue_vild_dir[category_name].get()
			top_k_item_clip = priority_queue_clip_dir[category_name].get()
			if not headless:
				axs[0, k].set_title(f"ViLD score {top_k_item_vild[0]*-1:.3f}")
				axs[0, k].imshow(top_k_item_vild[1][2])

				axs[1, k].set_title(f"CLIP score {top_k_item_clip[0]*-1:.3f}")
				axs[1, k].imshow(top_k_item_clip[1][2])

			#### Point cloud stuff
			#### Just show CLIP for now!
			file_num = int(top_k_item_clip[1][0].split("_")[1].split(".")[0])
			depth_img = pickle.load(open(img_dir_root_path+img_dir_name+"/depth_"+str(file_num),"rb"))
			rotation_matrix = pose_dir[file_num]['rotation_matrix']
			position = pose_dir[file_num]['position']

			ymin, xmin, ymax, xmax = top_k_item_clip[1][3:]

			center_y = int((ymin + ymax)/2.0)
			center_x = int((xmin + xmax)/2.0)

			transformed_point,bad_point = pixel_to_vision_frame(center_y,center_x,depth_img,rotation_matrix,position)
			side_pointx,_ = pixel_to_vision_frame_depth_provided(center_y,xmax,depth_img[center_y,center_x],rotation_matrix,position)
			side_pointy,_ = pixel_to_vision_frame_depth_provided(ymax,center_x,depth_img[center_y,center_x],rotation_matrix,position)

			#TODO: what should bb_size be for z? Right now, just making it same as x. Also needs to be axis aligned
			bb_sizex = np.linalg.norm(transformed_point-side_pointx)[0]*2
			bb_sizey = np.linalg.norm(transformed_point-side_pointy)[0]*2
			


			if bad_point:
				print(f"0 depth at the point for item {k} next bounding box")
			else:
				if type(best_pose) == type(None):
					best_pose = transformed_point
				print(f"item {k} good inside {top_k_item_clip[1][0]}")
				print(transformed_point)
				bb = o3d.geometry.OrientedBoundingBox(center=np.array(transformed_point),R=np.array([[1,0,0],[0,1,0],[0,0,1]]), extent=np.array([bb_sizex,bb_sizex,bb_sizey]))
				bb.color = [1,0,0]
				axis_center = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5,origin=transformed_point)
				top_axes.append(bb)
				top_axes.append(axis_center)

		if not headless:
			plt.show()
		o3d.visualization.draw_geometries([pcd]+top_axes)
		if type(best_pose) != type(None):
			fig, axs = plt.subplots(1, 2)
			move_to(robot,robot_state_client,pose=best_pose)

			# Capture and save images to disk
			image_responses = image_client.get_image_from_sources(sources)

			cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

			cv2.imwrite("./tmp/color_curview_.jpg", cv_visual)

			priority_queue_vild_dir_cur, priority_queue_clip_dir_cur = get_best_clip_vild_dirs(["color_curview_.jpg"],"./tmp",cache_images=False,cache_text=False,cache_path=cache_path,img_dir_name="",category_names=[category_name],headless=headless)

			#TODO: For now, just get top region and pick
			top_k_item_vild = priority_queue_vild_dir_cur[category_name].get()
			top_k_item_clip = priority_queue_clip_dir_cur[category_name].get()

			ymin, xmin, ymax, xmax = top_k_item_clip[1][3:]

			center_y = int((ymin + ymax)/2.0)
			center_x = int((xmin + xmax)/2.0)

			best_pixel = (center_x, center_y)

			print(best_pixel)

			axs[0].imshow(cv_visual)
			axs[1].imshow(top_k_item_clip[1][2])
			plt.savefig('./tmp/crops.png')
			#plt.show()

			arm_object_grasp(robot_state_client,manipulation_api_client,best_pixel,image_responses[1])


