import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
import time


def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = "Robot is estopped. Please use an external E-Stop client, such as the" \
        " estop SDK example, to configure E-Stop."
        robot.logger.error(error_message)
        raise Exception(error_message)

def open_gripper(robot_command_client):
    gripper_open = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
    cmd_id = robot_command_client.robot_command(gripper_open)


def arm_object_grasp(robot_state_client, manipulation_api_client, pick_pixels,image):
    pick_vec = geometry_pb2.Vec2(x=pick_pixels[0], y=pick_pixels[1])

    # Build the proto
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole)

    # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
    add_grasp_constraint(grasp, robot_state_client)

    # Ask the robot to pick up the object
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

    # Send the request
    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=grasp_request)

    # Get feedback from the robot
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        print('Current state: ',
              manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

        if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
            break

        time.sleep(0.25)


    time.sleep(4.0)

def add_grasp_constraint(grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = False
    force_45_angle_grasp = False
    force_squeeze_grasp = False

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if config.force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def pixel_to_vision_frame_depth_provided(i,j,depth,rotation_matrix,position):
	'''
	Converts a pixel (i,j) in HxW image to 3d position in vision frame

	i,j: pixel location in image
	depth_img: HxW depth image
	rotaton_matrix: 3x3 rotation matrix of hand in vision frame
	position: 3x1 position vector of hand in vision frame
	'''

	#hand_tform_camera comes from line below, just a hardcoded version of it
	#rot2 = mesh_frame.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))
	hand_tform_camera = np.array([[ 3.74939946e-33,6.12323400e-17,1.00000000e+00],
	[-1.00000000e+00,6.12323400e-17,0.00000000e+00],
	[-6.12323400e-17,-1.00000000e+00,6.12323400e-17]])

	#Intrinsics for RGB hand camera on spot
	CX = 320
	CY = 240
	FX= 552.0291012161067
	FY = 552.0291012161067


	z_RGB = depth
	x_RGB = (j - CX) * z_RGB / FX
	y_RGB = (i - CY) * z_RGB / FY

	bad_z = z_RGB == 0 #if z_RGB is 0, the depth was 0, which means we didn't get a real point. x,y,z will just be where robot hand was

	#first apply rot2 to move camera into hand frame, then apply rotation + transform of hand frame in vision frame
	transformed_xyz = np.matmul(rotation_matrix,np.matmul(hand_tform_camera,np.array([x_RGB,y_RGB,z_RGB]))) + position

	return(transformed_xyz,bad_z)


def pixel_to_vision_frame(i,j,depth_img,rotation_matrix,position):
	'''
	Converts a pixel (i,j) in HxW image to 3d position in vision frame

	i,j: pixel location in image
	depth_img: HxW depth image
	rotaton_matrix: 3x3 rotation matrix of hand in vision frame
	position: 3x1 position vector of hand in vision frame
	'''

	#hand_tform_camera comes from line below, just a hardcoded version of it
	#rot2 = mesh_frame.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))
	hand_tform_camera = np.array([[ 3.74939946e-33,6.12323400e-17,1.00000000e+00],
	[-1.00000000e+00,6.12323400e-17,0.00000000e+00],
	[-6.12323400e-17,-1.00000000e+00,6.12323400e-17]])

	#Intrinsics for RGB hand camera on spot
	CX = 320
	CY = 240
	FX= 552.0291012161067
	FY = 552.0291012161067


	z_RGB = depth_img[i,j]
	x_RGB = (j - CX) * z_RGB / FX
	y_RGB = (i - CY) * z_RGB / FY

	bad_z = z_RGB == 0 #if z_RGB is 0, the depth was 0, which means we didn't get a real point. x,y,z will just be where robot hand was

	#first apply rot2 to move camera into hand frame, then apply rotation + transform of hand frame in vision frame
	transformed_xyz = np.matmul(rotation_matrix,np.matmul(hand_tform_camera,np.array([x_RGB,y_RGB,z_RGB]))) + position

	return(transformed_xyz,bad_z)
