from collections import defaultdict
from tqdm import tqdm
import os
import pickle
import numpy as np

from vild.vild_utils import *
import math
from queue import PriorityQueue
import open3d as o3d

def get_best_clip_vild_dirs(model,preprocess,img_names,img_dir_path,use_softmax=False,cache_images=True,cache_text=True,cache_path="./cache/",img_dir_name=None,category_names=None,headless=False):
	#################################################################
	#Lots of hyperparameters, make more general TODO
	overall_fig_size = (18, 24)

	max_boxes_to_draw = 25 #@param {type:"integer"}
	nms_threshold = 0.6 #@param {type:"slider", min:0, max:0.9, step:0.05}
	min_rpn_score_thresh = 0.9  #@param {type:"slider", min:0, max:1, step:0.01}
	min_box_area = 220 #@param {type:"slider", min:0, max:10000, step:1.0}
	#min_box_area = 1000
	params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area

	mask_color =   'red'
	alpha = 0.5
	fig_size_w = 35
	line_thickness = 2

	saved_model_dir = "./vild/image_path_v2"

	#################################################################
	# load model for clip image embeddings on crops
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Models: {clip.available_models()}")
	model, preprocess = clip.load("ViT-B/32", device=device)

	#################################################################
	# text stuff
	categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
	category_indices = {cat['id']: cat for cat in categories}

	#################################################################
	# Compute text embeddings with CLIP
	cache_text_exists = os.path.isfile(cache_path+img_dir_name+"_text")
	if cache_text and cache_text_exists:
			text_features = pickle.load(open(cache_path+img_dir_name+"_text","rb"))
	else:
			text_features = build_text_embedding(categories,model,preprocess)
			pickle.dump(text_features,open(cache_path+img_dir_name+"_text","wb"))

	#################################################################
	# check if cache exists
	cache_img_exists = os.path.isfile(cache_path+img_dir_name+"_images_vild")
	# if cache exist, use it to load in image vectors from img names
	if cache_images:
		if cache_img_exists:
			img2vectorvild_dir = pickle.load(open(cache_path+img_dir_name+"_images_vild","rb"))
			img2vectorclip_dir = pickle.load(open(cache_path+img_dir_name+"_images_clip","rb"))
		else:
			img2vectorvild_dir = {}
			img2vectorclip_dir = {}
		session = None
	if not cache_img_exists or not cache_images:
		# Load ViLD model
		session = tf.Session(graph=tf.Graph())
		_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)

	#################################################################

	priority_queue_clip_dir = defaultdict(lambda: PriorityQueue()) #keys will be category names. The priority will be negative score (since lowest gets dequeue) and items be image, anno_idx, and crop
	priority_queue_vild_dir = defaultdict(lambda: PriorityQueue()) #keys will be category names. The priority will be negative score (since lowest gets dequeue) and items be image, anno_idx, and crop
	for img_name in tqdm(img_names):
		image_path = img_dir_path + "/" + img_name
		#print(image_path)

		###################################################
		#Load all of the images, and use ViLD to extract RoI bounding boxes (+rescaled), masks, scores, and feature vectors.
		if cache_images and cache_img_exists:
			image,image_height,image_width,valid_indices,detection_roi_scores,detection_boxes,detection_masks,detection_visual_feat,rescaled_detection_boxes = img2vectorvild_dir[img_name]	
		else:
			image,image_height,image_width,valid_indices,detection_roi_scores,detection_boxes,detection_masks,detection_visual_feat,rescaled_detection_boxes  = extract_roi_vild(image_path,session,params)
		if cache_images and not cache_img_exists:
			img2vectorvild_dir[img_name] = [image,image_height,image_width,valid_indices,detection_roi_scores,detection_boxes,detection_masks,detection_visual_feat,rescaled_detection_boxes]
		

		#################################################################
		# Compute detection scores, and rank results
		#print(f"Detection visual feat {detection_visual_feat.shape}")
		#print(f"text feat {text_features.shape}")
		raw_scores = detection_visual_feat.dot(text_features.T)
		#print(raw_scores)
		if use_softmax:
			scores_all = softmax(temperature * raw_scores, axis=-1)
		else:
			scores_all = raw_scores

		indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
		indices_fg = indices #I've replaced the line below with this because original code base treated first idx as background and removed it, not relevant here
		#indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])

		#################################################################
		# Plot detected boxes on the input image.
		ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
		processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
		segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

		if len(indices_fg) == 0 and not headless:
			display_image(np.array(image), size=overall_fig_size)
			print('ViLD does not detect anything belong to the given category')

		else:
			if not headless:
				image_with_detections = visualize_boxes_and_labels_on_image_array(
				    np.array(image),
				    rescaled_detection_boxes[indices_fg],
				    valid_indices[:max_boxes_to_draw][indices_fg],
				    detection_roi_scores[indices_fg],    
				    numbered_category_indices,
				    instance_masks=segmentations[indices_fg],
				    use_normalized_coordinates=False,
				    max_boxes_to_draw=max_boxes_to_draw,
				    min_score_thresh=min_rpn_score_thresh,
				    skip_scores=False,
				    skip_labels=True)

		if not headless and vis_boxes:
			plt.figure(figsize=overall_fig_size)
			plt.imshow(image_with_detections)
			plt.axis('off')
			plt.title('Detected objects and RPN scores')
			plt.show()


		#################################################################
		# Plot
		cnt = 0
		raw_image = np.array(image)

		n_boxes = rescaled_detection_boxes.shape[0]

		if cache_images and not cache_img_exists:
			img2vectorclip_dir[img_name] = {}

		for anno_idx in indices[0:int(n_boxes)]:
			#print(img_name, anno_idx)
			rpn_score = detection_roi_scores[anno_idx]
			bbox = rescaled_detection_boxes[anno_idx]
			scores = scores_all[anno_idx]

			y1, x1, y2, x2 = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
			crop = np.copy(raw_image[y1:y2, x1:x2, :])

			for idx, category_name in enumerate(category_names):
				new_item = (-scores[idx], (img_name,anno_idx,crop,ymin[anno_idx],xmin[anno_idx],ymax[anno_idx],xmax[anno_idx]))
				#print(category_name, img_name, anno_idx)
				if new_item in priority_queue_vild_dir[category_name].queue:
					raise Exception(f"{img_name} {anno_idx} already in queue for {category_name}")
				priority_queue_vild_dir[category_name].put(new_item) #TODO: make this an object to more interpretable


			#crop_processed = np.moveaxis(crop, -1, 0)
			crop_pil = Image.fromarray(crop)
			if cache_images and not cache_img_exists:
				crop_pil.save("./cache/"+img_dir_name+"_crop.jpeg")
				crop_back = Image.open("./cache/"+img_dir_name+"_crop.jpeg")
				crop_processed = preprocess(crop_back).unsqueeze(0).to(device)
				clip_image_features = model.encode_image(crop_processed)

				img2vectorclip_dir[img_name][anno_idx] = clip_image_features

			elif cache_images:
				clip_image_features = img2vectorclip_dir[img_name][anno_idx]
			else:
				crop_pil.save("./cache/"+img_dir_name+"_crop.jpeg")
				crop_back = Image.open("./cache/"+img_dir_name+"_crop.jpeg")
				crop_processed = preprocess(crop_back).unsqueeze(0).to(device)
				clip_image_features = model.encode_image(crop_processed)

			#Normalize clip_image_features before taking dot product with normalized text features
			clip_image_features = clip_image_features / clip_image_features.norm(dim=1, keepdim=True)
			clip_image_features = clip_image_features.cpu().detach().numpy()
			clip_scores = clip_image_features.dot(text_features.T)

			#go through each word in the categories:
			for idx, category_name in enumerate(category_names):
				priority_queue_clip_dir[category_name].put((-clip_scores[0][idx], (img_name,anno_idx,crop,ymin[anno_idx],xmin[anno_idx],ymax[anno_idx],xmax[anno_idx]))) #TODO: make this an object to more interpretable



			img_w_mask = plot_mask(mask_color, alpha, raw_image, segmentations[anno_idx])
			crop_w_mask = img_w_mask[y1:y2, x1:x2, :]


			if not headless and vis_details:
				fig, axs = plt.subplots(1, 4, figsize=(fig_size_w, fig_size_h), gridspec_kw={'width_ratios': [3, 1, 1, 2]}, constrained_layout=True)

				# Draw bounding box.
				rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=line_thickness, edgecolor='r', facecolor='none')
				axs[0].add_patch(rect)

				axs[0].set_xticks([])
				axs[0].set_yticks([])
				axs[0].set_title(f'bbox: {y1, x1, y2, x2} area: {(y2 - y1) * (x2 - x1)} rpn score: {rpn_score:.4f}')
				if not headless:
					axs[0].imshow(raw_image)

				# Draw image in a cropped region.
				axs[1].set_xticks([])
				axs[1].set_yticks([])

				axs[1].set_title(f'predicted: {category_names[np.argmax(scores)]}')
				if not headless:
					axs[1].imshow(crop)

				# Draw segmentation inside a cropped region.
				axs[2].set_xticks([])
				axs[2].set_yticks([])
				axs[2].set_title('mask')
				if not headless:
					axs[2].imshow(crop_w_mask)

				# Draw category scores.
				fontsize = max(min(fig_size_h / float(len(category_names)) * 45, 20), 8)
				for cat_idx in range(len(category_names)):
				  axs[3].barh(cat_idx, scores[cat_idx], 
				              color='orange' if scores[cat_idx] == max(scores) else 'blue')
				axs[3].invert_yaxis()
				axs[3].set_axisbelow(True)
				axs[3].set_xlim(0, 1)
				plt.xlabel("confidence score")
				axs[3].set_yticks(range(len(category_names)))
				axs[3].set_yticklabels(category_names, fontdict={
				    'fontsize': fontsize})
				plt.show()
			cnt += 1
			# fig.tight_layout()


		#print('Detection counts:', cnt)
		#plt.show()
	if cache_images and not cache_img_exists:
		pickle.dump(img2vectorvild_dir,open(cache_path+img_dir_name+"_images_vild","wb"))
		pickle.dump(img2vectorclip_dir,open(cache_path+img_dir_name+"_images_clip","wb"))

	return(priority_queue_vild_dir, priority_queue_clip_dir)