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