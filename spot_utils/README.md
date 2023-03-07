This is a set of utility functions that are useful for interfacing with the spot.

## Camera Intrinsics
Getting the camera intrinsics is important for relating image coordinates to world coordinates (within the camera frame). You can use

`python3 get_intrinsics.py ROBOT_IP --image-sources IMAGE_SOURCE`

And the focal length, principal points, and skew should print out. For example, you can run

`python3 get_intrinsics.py 138.16.161.12 --image-sources hand_color_image`

for the spot rgb hand camera, the focal length for x and y may be 552.0291012161067, and principal x,y may be (320, 240), 0 skew. The relevant intrinsics matrix would be:

`intrinsics_matrix = np.array([[552.0291012161067, 0 ,320.0],[0,552.0291012161067,240.0],[0,0,1]])`

## Collect depth + color images and robot poses
You can teleoperate the spot using the controller, and then run the following program:

`python get_depth_color_pose.py --manual_images [True/False] --path_dir [DIR_PATH] --dir_name [DIR_NAME] [ROBOT_IP]`

For example:

`python get_depth_color_pose.py --manual_images True --path_dir ../data --dir_name spot_data 138.16.161.12`

I suggest making sure the robot hand is open so that the depth camera is blocked less. You will be prompted to take a photo all the time. Move the robot where you want and then hit enter. The program will capture the depth and color image, convert the depth image to be in color image space, and then get the robot pose in the vision frame. All of this is then saved to a folder. If you'd prefer the robot to continuously take images at a fixed frequency, pass False to the manual_images flag. path_dir is the path to where the new data directory will be located, and dir_name is the name of the new directory for the data.

When manual_images is False, the robot will only take photos after it has moved and is currently still (to reduce the number of blurry and unneeded photos). If high precision is required, use manual mode and take images when the robot is still.

## Generate pointcloud from data
After collecting depth, color and pose data, we can generate a point cloud of the scene, along with axes representing the pose of the robot's hand in the vision frame. In addition, we can save the pointcloud file in the same location we have the sensor data

`python generate_pointcloud.py`

## Move-to Location
If you want to stand the robot up and move 2 meters within a given a location in the vision frame, you can check out 

`python move_spot_to.py [ROBOT_IP]`

## E-stop
You'll need to run an e-stop if you want the robot to operate autonomously, I suggest looking at the example code from the Spot SDK:

https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples/estop

## TODO
- make generate point cloud way more effecient by using transform with position and rotation data on pcd rather than add apply to each point individually
- get_depth_color_pose.py should have options for visual vs. map frame
- arguments should be passed in at command line instead of hard coded
- move_spot_to has temporary "stupid" class, and should have adjustable approach range, and also handle obstacles, and also connect to seed frame
- timeout for move_spot_to should be considered more, also make navigation more advance (todo above)

## BUGS 
- (?) I am using the hand frame for get_depth_color_pose.py but I should double check that this is aligned with hand color frame
	- Seems like it's not the same exact frame, but there is an off-set that is calculated in visualize_data.py currently, move it somewhere else