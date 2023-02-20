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

`python get_depth_color_pose.py ROBOT_IP`

For example:

`python get_depth_color_pose.py 138.16.161.12`

You will be prompted to take a photo. Move the robot where you want and then hit enter. The program will capture the depth and color image, convert the depth image to be in color image space, and then get the robot pose in the vision frame. All of this is then saved to a folder.

## TODO
- get_depth_color_pose.py should have options for visual vs. map frame
- arguments should be passed in at command line instead of hard coded

## BUGS 
- (?) I am using the hand frame for get_depth_color_pose.py but I should double check that this is aligned with hand color frame
	- Seems like it's not the same exact frame, but there is an off-set that is calculated in visualize_data.py currently, move it somewhere else