This is a repository for an implementation of nlmap-saycan, with additional utilities for the Spot.

nlmap-saycan: https://nlmap-saycan.github.io/

## Setup
If you haven't yet, make a virtual environment and activate it, for example with conda:
`conda create -n nlmap_spot python=3.9`
`conda activate nlmap_spot`

Clone this repo:
`git clone git@github.com:ericrosenbrown/nlmap_spot.git`

Setup CLIP and ViLD based on the following links:

CLIP: https://github.com/openai/CLIP

ViLD: https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/vild/ViLD_demo.ipynb

## Usage
TODO: explain flags (caching)
TODO: 

## Examples
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

## TODO:
- make data collection on spot have continuous mode (maybe even autonomous?)
- clean out unneeded print statements
- Filter out low-score candidates
- Make configuration file for language and other hyperparameters and load them in
- decide on structure for how color, depth, and pose is stored
- don't use center pixel of bounding box, instead use "center" of mask
- Make semantic map as in NL-Map
	- run classify_top_k on rgb images
	- use bounding boxes to get estimated size of objects
	- visualize object locations in map too!
	- do multi-view semantic fusion
- Hook in LLM to reproduce nlmap-saycan
- Write utilies to collect data on Spot + execute navigation and manipulation actions with nlmap
- Set up distributed processing so entire pipeline can happen in parallel and on robot platform
- handle LTL specifications for navigation (globally, until, etc.) by using nlmap to genrate invalid/required locations, path planning accordingly.


## BUGS:
- (?) Why do CLIP scores seem to always be slightly lower than ViLD, is it due to do normalizaton issues?
- in pointcloud_classify_all.py etc. there is a window error when trying to visualize matplotlib and open3d? Seems like I can only do one for some reason, so currently needs to be run in headless mode
