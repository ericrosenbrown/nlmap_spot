from vild.vild_utils import *
import pickle
import os
import open3d as o3d
from spot_utils.utils import pixel_to_vision_frame

#################################################################
# Hyperparameters and general initialization
cache_images = True #load image cache if available, make image cache when needed
cache_text = True#same but for text
vis_boxes = True #show image with detected bounding boxes
vis_details = True #show details for each bounding box
headless = False
img_dir_root_path = "./data/"
img_dir_name = "spot-depth-color-pose-data3"
img_dir_path = img_dir_root_path + img_dir_name
cache_path = "./cache/"
saved_model_dir = "./vild/image_path_v2"
pose_data_fname = "pose_data.pkl"

overall_fig_size = (18, 24)

max_boxes_to_draw = 25 #@param {type:"integer"}
nms_threshold = 0.6 #@param {type:"slider", min:0, max:0.9, step:0.05}
min_rpn_score_thresh = 0.9  #@param {type:"slider", min:0, max:1, step:0.01}
min_box_area = 220 #@param {type:"slider", min:0, max:10000, step:1.0}
#min_box_area = 1000
params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area

use_softmax = False

mask_color =   'red'
alpha = 0.5
fig_size_w = 35
line_thickness = 2

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
#category_name_string = "Table; Chair; Sofa; Lamp; Rug; Television; Fireplace; Pillow; Blanket; Clock; Picture frame; Vase; Lampshade; Candlestick; Books; Magazines; DVD player; CD player; Record player; Video game console; Board game; Card game; Chess set; Backgammon set; Carpet; Drapes; Blinds; Shelving unit; Side table; Coffee table; Footstool; Armchair; Bean bag; Desk; Office chair; Computer; Printer; Scanner; Fax machine; Telephone; Cell phone; Lamp; Lamp; Rug; Trash can; Wastebasket; Vacuum cleaner; Broom; Dustpan; Mop; Bucket; Dust cloth; Cleaning supplies; Iron; Ironing board; Hair dryer; Curling iron; Toilet brush; Towels; Soap; Shampoo; Toothbrush; Toothpaste; Razor; Shaving cream; Deodorant; Hairbrush; Hair ties; Makeup; Nail polish; Perfume; Cologne; Laundry basket; Clothes hanger; Closet; Dresser; Bed; Mattress; Pillows; Sheets; Blanket; Comforter; Quilt; Bedspread; Nightstand; Alarm clock; Lamp; Lamp; Rug"
#category_name_string = "Table; Chair"
category_name_string = "Hairbrush; Lamp; Chair; Sofa; Books; Television" 


category_names = [x.strip() for x in category_name_string.split(';')]
#category_names = ['background'] + category_names
categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
category_indices = {cat['id']: cat for cat in categories}

fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)

#################################################################
# Compute text embeddings with CLIP
cache_text_exists = os.path.isfile(cache_path+img_dir_name+"_text")
if cache_text and cache_text_exists:
		text_features = pickle.load(open(cache_path+img_dir_name+"_text","rb"))
else:
		text_features = build_text_embedding(categories)
		pickle.dump(text_features,open(cache_path+img_dir_name+"_text","wb"))

#################################################################
# load model for clip image embeddings on crops
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Models: {clip.available_models()}")
model, preprocess = clip.load("ViT-B/32", device=device)
#################################################################
# Loop through images at path locations
img_names = os.listdir(img_dir_path)
img_names = [img_name for img_name in img_names if "color" in img_name]
img_names.sort()

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
if not cache_img_exists:
	# Load ViLD model
	session = tf.Session(graph=tf.Graph())
	_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)
for img_name in img_names:
	image_path = img_dir_path + "/" + img_name

	###################################################
	#Load all of the images, and use ViLD to extract RoI bounding boxes (+rescaled), masks, scores, and feature vectors.
	if cache_images and cache_img_exists:
		image,image_height,image_width,valid_indices,detection_roi_scores,detection_boxes,detection_masks,detection_visual_feat,rescaled_detection_boxes = img2vectorvild_dir[img_name]	
	else:
		image,image_height,image_width,valid_indices,detection_roi_scores,detection_boxes,detection_masks,detection_visual_feat,rescaled_detection_boxes  = extract_roi_vild(image_path,session,overall_fig_size,params)
	if cache_images and not cache_img_exists:
		img2vectorvild_dir[img_name] = [image,image_height,image_width,valid_indices,detection_roi_scores,detection_boxes,detection_masks,detection_visual_feat,rescaled_detection_boxes]
	

	#################################################################
	# Compute detection scores, and rank results
	raw_scores = detection_visual_feat.dot(text_features.T)
	if use_softmax:
		scores_all = softmax(temperature * raw_scores, axis=-1)
	else:
		scores_all = raw_scores

	indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
	indices_fg = indices#np.array([i for i in indices if np.argmax(scores_all[i]) != 0])

	#################################################################
	# Plot detected boxes on the input image.
	ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
	processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
	segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

	if len(indices_fg) == 0:
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
		print(img_name, anno_idx)
		rpn_score = detection_roi_scores[anno_idx]
		bbox = rescaled_detection_boxes[anno_idx]
		scores = scores_all[anno_idx]

		y1, x1, y2, x2 = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
		crop = np.copy(raw_image[y1:y2, x1:x2, :])

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

		center_y = int((ymin[anno_idx] + ymax[anno_idx])/2.0)
		center_x = int((xmin[anno_idx] + xmax[anno_idx])/2.0)

		#### Point cloud visualization
		file_num = int(img_name.split("_")[-1].split(".")[0])
		depth_img = pickle.load(open(img_dir_root_path+img_dir_name+"/depth_"+str(file_num),"rb"))
		rotation_matrix = pose_dir[file_num]['rotation_matrix']
		position = pose_dir[file_num]['position']
		transformed_point,bad_point = pixel_to_vision_frame(center_y,center_x,depth_img,rotation_matrix,position)

		if bad_point:
			print("0 depth at the point, next bounding box")
		else:
			#bb = o3d.geometry.OrientedBoundingBox(center=np.array(transformed_point),R=np.array([[1,0,0],[0,1,0],[0,0,1]]), extent=np.array([1,1,1]))
			bb= o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=transformed_point)
			o3d.visualization.draw_geometries([pcd,bb])


	print('Detection counts:', cnt)
	#plt.show()
if cache_images and not cache_img_exists:
	pickle.dump(img2vectorvild_dir,open(cache_path+img_dir_name+"_images_vild","wb"))
	pickle.dump(img2vectorclip_dir,open(cache_path+img_dir_name+"_images_clip","wb"))

