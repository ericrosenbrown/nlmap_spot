This is a repository for an implementation of nlmap-saycan, with additional utilities for the Spot.

nlmap-saycan: https://nlmap-saycan.github.io/

# Setup
If you haven't yet, make a virtual environment and activate it, for example with conda:
`conda create -n nlmap_spot python=3.9`
`conda activate nlmap_spot`

Clone this repo:
`git clone git@github.com:ericrosenbrown/nlmap_spot.git`

Setup CLIP and ViLD based on the following links:

CLIP: https://github.com/openai/CLIP

ViLD: https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/vild/ViLD_demo.ipynb

The codebase is configured so that if you have an offline dataset you wish to process, you can do so without connecting to a Spot at all (explained in Usage section). If you want to connect to the Spot, you'll be asked to supply a username and password. While you can do this manually each time you run the program, you can also just store the username and password into the environmental variables BOSDYN_CLIENT_USERNAME and BOSDYN_CLIENT_PASSWORD respectively.

# Usage
This codebase is configued to offer various functionalities, primarily:
1. Collecting RGB-D and pose data from the Spot
2. Processing collected data into a colored pointcloud
3. Construct a queryable scene representation based on data from (1) (i.e: make an NLMap)
4. Given a natural language query, visualize the top K results  (either as 2D images or in 3D pointcloud)
5. Given a natural language query, query NLMap for pose and navigate robot to location + pick object
6. Given natural language task, use LLM to generate relevant objects

The only functionalities that require connecting to a real robot are for collecting data (1) and having the robot navigatge to objects and pick them up (5). All other functionalities can be executed offline without connecting to a real robot as long as there exists a dataset from (1) already saved. **If you want to use this code base without connecting to a robot and just process data offline (i.e: you don't need to do (1) or (5), then set use_robot=False in config file (more information below). No robot information will be needed or used).** We will now walkthrough each of these functionalities.

Besides Collecting RGB-D and pose data from the Spot (1), all of the other functionalities are done through the `nlmap.py` script. `nlmap.py` takes in a single argument, a path to a configuration file (more details on the configuration file are provided below):

`python nlmap.py --config_path [PATH_TO_CONFIG]`

If no config_path is provided, a default config file will be used (located in `./configs/example.ini`)

`nlmap.py` contains an NLMap class, and calling this script simply makes an NLMap object based on the config file. Then, there are various methods that can be called from the instantiated object, which we describe in more detail. 

## (1) Collecting RGB-D and pose data from the Spot
Before anything, we need to collect sensor data of the scene using the Spot. Go into the spot_utils folder, this contains all the scripts that are specific to Spot. There is a README that describes some of the various helper scripts. Read the section **Collect depth + color images and robot poses**, this will provide information on how to collect data on the Spot. 

Running this code will save all of the data to a folder depending on the path_dir and dir_name provided. Every ith time sensor data is collected, the RGB image will be stored as color_i.jpg, the depth data will be stored as a pickle file as depth_i, and a visualized combination of the depth data on the RGB image will be stored as combined_i.jpg. In addition, there will be a pickle file called pose_data.pkl, this file contains a dictionary  whose keys are i, and whose values are a dictionary with keys 'position', 'quaternion(wxyz)', 'rotation_matrix', 'rpy'. Currently, these are 3D position and rotation representations of where the hand_camera was in space when the image i was collected.

If you don't have access to a Spot and unable to collect data yourself, you can go to this Google drive link and download some already-collected datasets: https://drive.google.com/drive/folders/1zPUWyU7L6PBMpOTdUIQV_KG6yMhS1-dz

For all the other functionalities that rely on `nlmap.py`, you will need to pass the location and name of the directory that contains all the data through the configuration file. Specifically, you will need to edit data (under **[dir_names]**) and `data_dir_root` (under **[paths]**). Look at the **Config** section below for more details.

## (2) Processing collected data into a colored pointcloud
We can use the `nlmap.py` script to process a colored pointcloud from the data generated in (1) and visualize the pointcloud. After a pointcloud is made for the first time, we will save the resulting pointcloud into the data directory as `pointcloud.pcd` so that next time we can just load in the pointcloud. 

In the config file (under the [pointcloud] section), set `use_pointcloud = True`. Now when we run `nlmap.py`, if the pointcloud hasn't been made before, it will be generated and saved. Now, you can use the NLMap method `viz_pointcloud()` to visualize the pointcloud (i.e: see the nlmap.py bottom-of-file as example comment)

## (3) Construct a queryable scene representation based on data from (1) (i.e: make an NLMap)
This will happen automatically when you construct the NLMap object, so as long as you pass in a valid config file, this will occur. Things to note are that since applying the visual-language models to all the sensor data takes the most amount of time for this entire process, in the config file there are two parameters under **[cache]**: `images` and `text`. If these are true, then the visual and text embeddings respectively will be saved into folders (location depends on value of `cache_dir` under **[paths]**), and will be used next time the code is ran. 

There are also various visualization options you can have on/off, listed under [viz]. These are described in more detail in the configuration setting.

## (4) Given a natural language query, visualize the top K results 
Once we have a NLMap object, you can do this by running the method `viz_top_k(viz_2d,viz_pointcloud)` (see bottom of `nlmap.py` for example). If viz_2d is True, then you will see the 2d figures of the top k best results. If pointcloud=True, then these bounding boxes will be visualized in 3D over the full pointcloud.

## (5) Given a natural language query, query NLMap for pose and navigate robot to location + pick object
Once we have a NLMap object, you can do this by running the method `go_to_and_pick_top_k(obj)` (see bottom of `nlmap.py` for example). obj needs to be one of the text queries in `category_name_string` in the config (since it will use the cache to get the text embedding).

## (6) Given natural language task, use LLM to generate relevant objects
Note: This part is not directly connected to the NLMap code yet, but can be easily hooked in. You can find the code in the file `saycan.py`. Given a task, it uses a prompt-engineering approach to use a LLM from OpenAI to propopse relevant objects to use to solve the task. These results can be plugged directly into `category_name_string` of a config file.

# Config
Here we describe the types and meanings of the configuration file. The config file has different sections (represented in **bold**), and each section contains a list of configurations and their associated values (listed below the bolded section). You can find an example of a config in example.ini: https://github.com/ericrosenbrown/nlmap_spot/blob/dev/configs/example.ini

**[dir_names]** - config values related to names of directories\
`data` - the name of the directory that contains data (should be generated from step (1))\
\
**[text]** - config values related to text input\
`category_name_string` - list of open-query words we will use in NLMap. Currently values are separated by ;. If you change this config value and have text=True for caching, delete your text cache so that it can be remade\
`prompt_engineering` - boolean determining whether prompt-engineering should be used or not for text embeddings of category_name_string\
\
**[robot]** - config values related to robot.\
`use_robot` - boolean determining whether robot is being used or not. If you are not connected to a Spot or just want to process data offline, set this to False. Otherwise True\
`hostname` - string representing hostname/IP or Spot. Only matters if use_robot is True.\
\
**[paths]** - config values related to the relevant paths\
`data_dir_root` - directory that contains data for nlmap (where [dir_names]data is located)\
`cache_dir` - path to where caches are stored, only relevant if cache_images/text is used\
`figs_dir` - path to where figures are saved (only relevant if [viz]save_whole_boxes or [viz]save_anno_boxes is True)\
`vild_dir` - path to where image_path_v2 is stored for vild\
\
**[file_names]** - config values related to name of data files\
`pose` - name of pose pickle data (generally is pose_data.pkl)\
`pointcloud` - name of pointcloud data (generally is pointcloud.pcd)\
\
**[cache]** - config values related to caching results\
`images` - boolean determining whether image embeddings from CLIP/ViLD are saved\
`text` - boolean determining whether text embeddings from CLIP are saved\
\
**[viz]** - config values related to visualizing\
`boxes` - boolean determining whether 2D bounding boxes for ViLD are visualized when creating NLMap object (only relevant if cache does not exist)\
`save_whole_boxes` - boolean determining whether 2D bounding boxes are saved into figs_dir when creating NLMap object\
`save_anno_boxes` - boolean determining whether distribution over category_name_string is sasved into figs_dir when creating NLMap object\
`mask_color` - string (representing color), color of 2d bounding box border\
`alkpha` - float, alpha of mask_color\
`overall_fig_size` - string (X,Y) representing size of the figures\
\
**[fusion]** - config valus related to multi-view fusion\
`top_k` - number of top images used for multi-view fusion based on NLMap score\
\
**[pointcloud]** - config values related to pointcloud generation\
`use_point` - boolean. if true and pointcloud doesn't exist, generate one, otherwise use existing one. If false, don't generate point cloud (more limited capaibility)\
\
**[pose]** - config values related to pose data\
`use_pose` - boolean determining whether pose data should be loaded or not\
\
**[vild]** - config values related to ViLD\
`max_boxes_to_draw` - integer representing how many bounding boxes we consider from ViLD\
`nms_threshold` - float determining Non-maximum Suppression (NMS) threshold\
`min_rpn_score` - float determing region proposal network score minimum\
`min_box_area` - int determining min area of acceptable bounding boxes\
\
**[clip]**\
`model` - string of the name of CLIP model\