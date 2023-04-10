This is a repository for an implementation of nlmap-saycan, with additional utilities for the Spot.

nlmap-saycan: https://nlmap-saycan.github.io/

### Setup
If you haven't yet, make a virtual environment and activate it, for example with conda:
`conda create -n nlmap_spot python=3.9`
`conda activate nlmap_spot`

Clone this repo:
`git clone git@github.com:ericrosenbrown/nlmap_spot.git`

Setup CLIP and ViLD based on the following links:

CLIP: https://github.com/openai/CLIP

ViLD: https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/vild/ViLD_demo.ipynb

The codebase is configured so that if you have an offline dataset you wish to process, you can do so without connecting to a Spot at all (explained in Usage section). If you want to connect to the Spot, you'll be asked to supply a username and password. While you can do this manually each time you run the program, you can also just store the username and password into the environmental variables BOSDYN_CLIENT_USERNAME and BOSDYN_CLIENT_PASSWORD respectively.

### Usage
This codebase is configued to offer various functionalities, primarily:
1. Collecting RGB-D and pose data from the Spot
2. Processing collected data into a colored pointcloud
3. Construct a queryable scene representation based on data from (1) (i.e: make an NLMap)
4. Given a natural language query, visualize the top K results  (either as 2D images or in 3D pointcloud)
5. Given a natural language query, query NLMap for pose and navigate robot to location + pick object
6. Given natural language task, use LLM to generate relevant objects

The only functionalities that require connecting to a real robot are for collecting data (1) and having the robot navigatge to objects and pick them up (5). All other functionalities can be executed offline without connecting to a real robot as long as there exists a dataset from (1) already saved.

### Examples
Go to this google link drive and download some example images from a Spot walking around a room
spot-images examples: https://drive.google.com/file/d/1TOj7Chu089YmJS_gy_A0AFWR9DXux-0G/view?usp=share_link

Now, run 
`python classify_all.py`

This will take all the images from spot-images, run ViLD on each to extract bounding boxes and image features, and apply CLIP image encoder to the bounding boxes. A list of strings are provided, which CLIP text features are created for. We then visualize the ViLD RoI bounding boxes, confidence scores for text classes, and masks. In this example, we also take dot product between CLIP image features and text features as well as ViLD image features and text features, which is used in full nlmap.

There are cache options to save CLIP image features and textures + ViLD image features. This way, it is faster to run next time. You can just delete the cache if you make any changes.

If you'd like to see where each of the bounding boxes are in the pointcloud, go to spot_utils, generate a pointcloud first, then you can run:

`python pointcloud_classify_all.py`

If you want to see the top k results for a query, then do:

`python classify_top_k.py`

This will find the top k crops for a sequence of strings across all images for both ViLD and CLIP and visualize them. 

If you'd like to see where each of the top k bounding boxes are in the pointcloud, generate the point cloud with spot_utils and run:

`python pointcloud_classify_top_k.py`

If you'd like to get some object proposals from a LLM, you can run

`python saycan.py`

This will do some in-context learning and propose a list of objects that may be relevant to the task specified.

If you'd like to have the robot move to different objects based on the detectiosn, you can run

`python nlmap.py`

### TODO:
- Write explanation for config file
- Right now, I only save top crops of vild according to text score similarity for clip, which means when i change text I may need to change vision cache. Make this more effecient by saving all the results of vild, computing clip for all of them, and for new text get different anno idxs.
- I can't show back-to-back open3d visualizations in viz_top_k because of weird window crashing problem?
- move pointcloud utils around? is it kind of a weird place (outside of nlamp.py?)
- make it so i can either viz or not viz when I run generate_pointcloud (right now it always visualizes the point cloud + poses on generation)
- add poses to function that visualizes pointcloud
- check out context prompt is affecting results 
- save fig results for top_k 
- get_best_clip_vild_dirs is hackily used in go_to (since new images need to be processed)
- Move lease outisde of go_to_and_pick_top_k so that the robot can always move around 


- make data collection on spot autonomous?
- clean out unneeded print statements
- axis aligned bounding boxes for pointcloud
- Filter out low-score candidates
- Make configuration file for language and other hyperparameters and load them in
- decide on structure for how color, depth, and pose is stored
- don't use center pixel of bounding box, instead use "center" of mask
- only using CLIP scores right now for nlmap, also integrate in ViLD scores
- Make semantic map as in NL-Map
	- do multi-view semantic fusion
- Hook in LLM to reproduce nlmap-saycan
- Set up distributed processing so entire pipeline can happen in parallel and on robot platform
- saycan.py should have in-context learning as a passed parameter, cleaned up to have better options, etc.
- handle LTL specifications for navigation (globally, until, etc.) by using nlmap to genrate invalid/required locations, path planning accordingly.
- extract_roi_vild in vild_utils only operates on image_paths (session.run takes it in), work on making it so I can pass image in directly?

### BUGS:
- (?) Why do CLIP scores seem to always be slightly lower than ViLD, is it due to do normalizaton issues?
- in pointcloud_classify_all.py etc. there is a window error when trying to visualize matplotlib and open3d? Seems like I can only do one for some reason, so currently needs to be run in headless mode
