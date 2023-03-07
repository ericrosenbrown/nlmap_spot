# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Example demonstrating capture of both visual and depth images and then overlaying them."""

import argparse

import sys
import os
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient

import cv2
import numpy as np


def main(argv):
    # Parse args

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--manual_images", help="Whether images are taken manually or continuously",default="True")
    parser.add_argument("--path_dir", help="path to directory where data is saved",default="../data")
    parser.add_argument("--dir_name", help="name of directory for where data is stored",default="default_data")
    options = parser.parse_args(argv)

    if options.manual_images == "True":
        manual_images = True
    else:
        already_took_photo = False
        manual_images = False
        robot_is_moving = False
        robot_take_pic = True
        last_robot_position = [0,0,0]


    visualize = False #everytime we capture image, vis the results
    robot_pose = True #get camera frame in vision frame when picture taken
    img_dir = options.path_dir + "/" + options.dir_name + "/"
    pic_hz = 2 #if manual_images is false, the hz at which images are automatically taken

    if visualize:
        from matplotlib import pyplot as plt
    if robot_pose:
        from bosdyn.client.frame_helpers import get_a_tform_b, get_frame_names
        import pickle

    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)


    #We only use the hand color
    sources = ['hand_depth_in_hand_color_frame', 'hand_color_image']

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_depth_plus_visual')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    counter = 0
    img_to_pose_dir = {} #takes in counter as key, and returns robot pose. saved in img_dir
    while True:
        if manual_images:
            response = input("Take image [y/n]")
            if response == "n":
                break
        else:
            time.sleep(1/float(pic_hz))

        # Capture and save images to disk
        image_responses = image_client.get_image_from_sources(sources)

        if robot_pose:
            frame_tree_snapshot = robot.get_frame_tree_snapshot()
            vision_tform_hand = get_a_tform_b(frame_tree_snapshot,"vision","hand")

            img_to_pose_dir[counter] = {"position": [vision_tform_hand.position.x,vision_tform_hand.position.y,vision_tform_hand.position.z], 
            "quaternion(wxyz)": [vision_tform_hand.rotation.w,vision_tform_hand.rotation.x,vision_tform_hand.rotation.y,vision_tform_hand.rotation.z],
            "rotation_matrix": vision_tform_hand.rotation.to_matrix(),
            "rpy": [vision_tform_hand.rotation.to_roll(),vision_tform_hand.rotation.to_pitch(),vision_tform_hand.rotation.to_yaw()]}
            
            pickle.dump(img_to_pose_dir,open(img_dir+"pose_data.pkl","wb"))

            robot_position = [vision_tform_hand.position.x,vision_tform_hand.position.y,vision_tform_hand.position.z]

        if not manual_images:
            if np.linalg.norm(np.array(robot_position) - np.array(last_robot_position)) > 0.2: #robot is moving around
                print("Robot is moving, no photos!")
                last_robot_position = robot_position
                already_took_photo = False
                continue
            else:
                last_robot_position = robot_position
                if already_took_photo: #don't take a phot if you already have
                    print("Robot still but already took photo")
                    continue
                else:
                    print("Robot still: NEW PHOTO!")
                    already_took_photo = True #going to take photo, don't next time



        # Image responses are in the same order as the requests.
        # Convert to opencv images.

        if len(image_responses) < 2:
            print('Error: failed to get images.')
            return False

        # Depth is a raw bytestream
        cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows,
                                    image_responses[0].shot.image.cols)

        #cv_depth is in millimeters, divide by 1000 to get it into meters
        cv_depth_meters = cv_depth / 1000.0

        # Visual is a JPEG
        cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

        #cv2.imwrite("color.jpg", cv_visual)

        # Convert the visual image from a single channel to RGB so we can add color
        visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(cv_visual, cv2.COLOR_GRAY2RGB)



        # Map depth ranges to color

        # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
        min_val = np.min(cv_depth)
        max_val = np.max(cv_depth)
        depth_range = max_val - min_val
        depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

        # Add the two images together.
        out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)

        cv2.imwrite(img_dir+"color_"+str(counter)+".jpg", cv_visual)
        pickle.dump(cv_depth_meters, open(img_dir+"depth_"+str(counter),"wb"))
        cv2.imwrite(img_dir+"combined_"+str(counter)+".jpg", out)
        counter += 1

        if visualize:
            fig, axs = plt.subplots(1, 3)
            axs[0].set_title("Depth (in meters)")
            axs[0].imshow(cv_depth_meters)
            axs[1].set_title("Color")
            axs[1].imshow(cv_visual)
            axs[2].set_title("Depth + Color")
            axs[2].imshow(out)
            plt.show()



if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
