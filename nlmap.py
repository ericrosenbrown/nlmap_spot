import configparser
import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
from PIL import Image
import clip
import torch

class NLMap():
	def __init__(self,config_path="./configs/example.ini"):
		###########################################################################################################
		######### Initialization

		### Extract config file 
		self.config = configparser.ConfigParser()
		self.config.sections()
		self.config.read(config_path)

		### device setting
		device = "cuda" if torch.cuda.is_available() else "cpu"

		### Visualization
		if self.config["viz"].getboolean("boxes"):
			from vild.vild_utils import visualize_boxes_and_labels_on_image_array, plot_mask
			import matplotlib.pyplot as plt
			from matplotlib import patches

		### Robot initializaton
		if self.config["robot"].getboolean("use_robot"):
			self.sdk = bosdyn.client.create_standard_sdk('NLMapSpot')
			self.robot = sdk.create_robot(self.config["robot"]["hostname"])
			bosdyn.client.util.authenticate(self.robot)

			# Time sync is necessary so that time-based filter requests can be converted
			self.robot.time_sync.wait_for_sync()

			assert not self.robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
			                                "such as the estop SDK example, to configure E-Stop."

			robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)

			self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
			self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
			self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
			self.robot_command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

			#We only use the hand color
			self.sources = ['hand_depth_in_hand_color_frame', 'hand_color_image']

		### Values set based on config parameters
		self.data_dir_path = f"{self.config['paths']['data_dir_root']}/{self.config['dir_names']['data']}"

		### Load pose data
		if self.config["pose"].getboolean("use_pose"):
			pose_path = f"{self.data_dir_path}/{self.config['file_names']['pose']}"
			try:
				self.pose_dir = pickle.load(open(pose_path,"rb"))
			except:
				raise Exception(f"use_pose is true but no pose data found at {pose_path}")

		### Pointcloud initialization
		if self.config["pointcloud"].getboolean("use_pointcloud"):
			import open3d as o3d

			pointcloud_path = f"{self.data_dir_path}/{self.config['file_names']['pointcloud']}"
			if os.path.isfile(pointcloud_path):
				self.pcd = o3d.io.read_point_cloud(pointcloud_path)
			else:
				raise Exception(f"use_pointcloud is true but {pointcloud_path} does not exist. Implement GENERATE POINTCLOUD")

			#o3d.visualization.draw_geometries([self.pcd])

		### Text initialization
		self.category_names = [x.strip() for x in self.config["text"]["category_name_string"].split(';')]
		self.categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(self.category_names)]
		self.category_indices = {cat['id']: cat for cat in self.categories}



		### Cache path
		self.cache_path = f"{self.config['paths']['cache_dir']}/{self.config['dir_names']['data']}"
	
		### Compute text embeddings with CLIP 
		self.cache_text_exists = os.path.isfile(f"{self.cache_path}_text")

		#if text cache should be used and it exists, load it
		if self.config["cache"].getboolean("text") and self.cache_text_exists:
			self.text_features = pickle.load(open(f"{self.cache_path}_text","rb"))
		else: #build new text embeddings
			from vild.vild_utils import build_text_embedding

			#load CLIP model
			self.clip_model, self.clip_preprocess = clip.load(self.config["clip"]["model"])

			self.text_features = build_text_embedding(self.categories,self.clip_model,self.clip_preprocess)

			if self.config["cache"].getboolean("text"): #save the text cache
				pickle.dump(self.text_features,open(f"{self.cache_path}_text","wb"))

		### Image initialization
		self.image_names = os.listdir(self.data_dir_path)
		self.image_names = [image_name for image_name in self.image_names if "color" in image_name]

		### Load cached image embeddings if they exist and are to be used, or make them otherwise
		self.cache_image_exists = os.path.isfile(f"{self.config['paths']['cache_dir']}/{self.config['dir_names']['data']}_images_vild")

		if self.config["cache"].getboolean("images") and self.cache_image_exists: #if image cache should be used and it exists, load it in
			self.image2vectorvild_dir = pickle.load(open(f"{self.cache_path}_images_vild","rb"))
			self.image2vectorclip_dir = pickle.load(open(f"{self.cache_path}_images_clip","rb"))

			self.topk_vild= pickle.load(open(f"{self.cache_path}_topk_vild","rb"))
			self.topk_clip = pickle.load(open(f"{self.cache_path}_topk_clip","rb"))
		else: #make image embeddings (either because you're not using cache, or because you don't have cache)
			self.priority_queue_clip_dir = defaultdict(PriorityQueue) #keys will be category names. The priority will be negative score (since lowest gets dequeue) and items be image, anno_idx, and crop
			self.priority_queue_vild_dir = defaultdict(PriorityQueue) #keys will be category names. The priority will be negative score (since lowest gets dequeue) and items be image, anno_idx, and crop

			# Load ViLD model
			import tensorflow.compat.v1 as tf
			from vild.vild_utils import extract_roi_vild, paste_instance_masks

			self.session = tf.Session(graph=tf.Graph())
			_ = tf.saved_model.loader.load(self.session, ['serve'], self.config["paths"]["vild_dir"])

			#load CLIP model
			self.clip_model, self.clip_preprocess = clip.load(self.config["clip"]["model"])

			params = self.config["vild"].getint("max_boxes_to_draw"),  self.config["vild"].getfloat("nms_threshold"),  self.config["vild"].getfloat("min_rpn_score_thresh"),  self.config["vild"].getfloat("min_box_area")

			print("Computing image embeddings")
			self.image2vectorvild_dir = {}
			self.image2vectorclip_dir = {}
			for image_name in tqdm(self.image_names):
				image_path = f"{self.data_dir_path}/{image_name}"

				image,image_height,image_width,valid_indices,detection_roi_scores,detection_boxes,detection_masks,detection_visual_feat,rescaled_detection_boxes  = extract_roi_vild(image_path,self.session,params)
				self.image2vectorvild_dir[image_name] = [image,image_height,image_width,valid_indices,detection_roi_scores,detection_boxes,detection_masks,detection_visual_feat,rescaled_detection_boxes]
			
				### We only compute CLIP embeddings for vild crops that have highest score
				### Compute detection scores, and rank results

				raw_scores = detection_visual_feat.dot(self.text_features.T)

				if self.config["vild"].getboolean("use_softmax"):
					scores_all = softmax(temperature * raw_scores, axis=-1)
				else:
					scores_all = raw_scores

				indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores

				ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
				processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
				segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

				overall_fig_size = [float(x) for x in self.config["viz"]["overall_fig_size"].split(",")]

				if self.config["viz"].getboolean("boxes"):
					if len(indices) == 0:
						display_image(np.array(image), size=overall_fig_size)
						print('ViLD does not detect anything belong to the given category')

					else:
						image_with_detections = visualize_boxes_and_labels_on_image_array(
						    np.array(image),
						    rescaled_detection_boxes[indices],
						    valid_indices[:self.config["vild"].getint("max_boxes_to_draw")][indices],
						    detection_roi_scores[indices],    
						    self.category_indices,
						    instance_masks=segmentations[indices],
						    use_normalized_coordinates=False,
						    max_boxes_to_draw=self.config["vild"].getint("max_boxes_to_draw"),
						    min_score_thresh=self.config["vild"].getfloat("min_rpn_score_thresh"),
						    skip_scores=False,
						    skip_labels=True)


					plt.figure(figsize=overall_fig_size)
					plt.imshow(image_with_detections)
					plt.axis('off')
					plt.title('Detected objects and RPN scores')
					plt.show()

				raw_image = np.array(image)
				n_boxes = rescaled_detection_boxes.shape[0]

				### image2vectorclip_dir[image_name] is a directory with annotations (crops) as keys
				self.image2vectorclip_dir[image_name] = {}

				### Go through the top crops
				for anno_idx in indices[0:int(n_boxes)]:
					rpn_score = detection_roi_scores[anno_idx]
					bbox = rescaled_detection_boxes[anno_idx]
					scores = scores_all[anno_idx]

					y1, x1, y2, x2 = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
					crop = np.copy(raw_image[y1:y2, x1:x2, :])

					### Add crop to priority queue for ranking scores for VILD
					for idx, category_name in enumerate(self.category_names):
						new_item = (-scores[idx], (image_name,anno_idx,crop,ymin[anno_idx],xmin[anno_idx],ymax[anno_idx],xmax[anno_idx]))
						#print(category_name, image_name, anno_idx)
						if new_item in self.priority_queue_vild_dir[category_name].queue:
							raise Exception(f"{image_name} {anno_idx} already in queue for {category_name}")
						self.priority_queue_vild_dir[category_name].put(new_item) #TODO: make this an object to more interpretable

					### Run CLIP vision model on crop
					crop_pil = Image.fromarray(crop)

					#if (use cache and cache does not exist) or (cache image does not exist), process the data
					if ((self.config["cache"].getboolean("images") and not self.cache_image_exists) or (not self.cache_image_exists)):
						crop_fname = f"{self.cache_path}_{self.config['dir_names']['data']}_crop.jpeg"
						crop_pil.save(crop_fname)
						crop_back = Image.open(crop_fname)
						crop_processed = self.clip_preprocess(crop_back).unsqueeze(0).to(device)
						clip_image_features = self.clip_model.encode_image(crop_processed)

						self.image2vectorclip_dir[image_name][anno_idx] = clip_image_features

					clip_image_features = self.image2vectorclip_dir[image_name][anno_idx]

					#Normalize clip_image_features before taking dot product with normalized text features
					clip_image_features = clip_image_features / clip_image_features.norm(dim=1, keepdim=True)
					clip_image_features = clip_image_features.cpu().detach().numpy()
					clip_scores = clip_image_features.dot(self.text_features.T)

					### Add crop to priority queue for ranking scores for CLIP
					for idx, category_name in enumerate(self.category_names):
						self.priority_queue_clip_dir[category_name].put((-clip_scores[0][idx], (image_name,anno_idx,crop,ymin[anno_idx],xmin[anno_idx],ymax[anno_idx],xmax[anno_idx]))) #TODO: make this an object to more interpretable

					# TODO: fig_size_w and h are a little hardcoded, make more general?
					fig_size_w = 35
					fig_size_h = min(max(5, int(len(self.category_names) / 2.5) ), 10)

					if self.config["viz"].getboolean("boxes"):
						img_w_mask = plot_mask(self.config["viz"]["mask_color"], self.config["viz"].getfloat("alpha"), raw_image, segmentations[anno_idx])
						crop_w_mask = img_w_mask[y1:y2, x1:x2, :]

						fig, axs = plt.subplots(1, 4, figsize=(fig_size_w, fig_size_h), gridspec_kw={'width_ratios': [3, 1, 1, 2]}, constrained_layout=True)

						# Draw bounding box.
						rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=self.config["viz"].getfloat("line_thickness"), edgecolor='r', facecolor='none')
						axs[0].add_patch(rect)

						axs[0].set_xticks([])
						axs[0].set_yticks([])
						axs[0].set_title(f'bbox: {y1, x1, y2, x2} area: {(y2 - y1) * (x2 - x1)} rpn score: {rpn_score:.4f}')

						axs[0].imshow(raw_image)

						# Draw image in a cropped region.
						axs[1].set_xticks([])
						axs[1].set_yticks([])

						axs[1].set_title(f'predicted: {self.category_names[np.argmax(scores)]}')

						axs[1].imshow(crop)

						# Draw segmentation inside a cropped region.
						axs[2].set_xticks([])
						axs[2].set_yticks([])
						axs[2].set_title('mask')

						axs[2].imshow(crop_w_mask)

						# Draw category scores.
						fontsize = max(min(fig_size_h / float(len(self.category_names)) * 45, 20), 8)
						for cat_idx in range(len(self.category_names)):
						  axs[3].barh(cat_idx, scores[cat_idx], 
						              color='orange' if scores[cat_idx] == max(scores) else 'blue')
						axs[3].invert_yaxis()
						axs[3].set_axisbelow(True)
						axs[3].set_xlim(0, 1)
						plt.xlabel("confidence score")
						axs[3].set_yticks(range(len(self.category_names)))
						axs[3].set_yticklabels(self.category_names, fontdict={
						    'fontsize': fontsize})
						plt.show()


			if self.config["cache"].getboolean("images"):
				pickle.dump(self.image2vectorvild_dir,open(f"{self.cache_path}_images_vild","wb"))
				pickle.dump(self.image2vectorclip_dir,open(f"{self.cache_path}_images_clip","wb"))

				### For priority queue, just get the top k results since PriorityQueue is not picklable
				topk_vild = []
				topk_clip = []
				for k in range(self.config["fusion"].getint("top_k")):
					top_k_item_vild = self.priority_queue_vild_dir[category_name].get()
					top_k_item_clip = self.priority_queue_clip_dir[category_name].get()

					topk_vild.append(top_k_item_vild)
					topk_clip.append(top_k_item_clip)

				pickle.dump(topk_vild,open(f"{self.cache_path}_topk_vild","wb"))
				pickle.dump(topk_clip,open(f"{self.cache_path}_topk_clip","wb"))


if __name__ == "__main__":
	### Parse arguments from command line
	parser = argparse.ArgumentParser()
	parser.add_argument("-c","--config_path", help="Path to config file", type=str, default="./configs/example.ini")
	args = parser.parse_args()

	nlmap = NLMap(args.config_path)
