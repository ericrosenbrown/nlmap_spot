import sys
import argparse
import time
import numpy as np

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import geometry_pb2
from bosdyn.api import basic_command_pb2

from bosdyn.client.frame_helpers import VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                        block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import math_helpers

class stupid:
    def __init__(self,x=3.80,y=2.743,z=0.564):
        self.x = x
        self.y = y
        self.z = z

def move_to(robot,robot_state_client,pose=None,distance_margin=1.25,hostname="138.16.161.12", end_time=30):
    robot.logger.info("Powering on robot... This may take a several seconds.")
    robot.power_on(timeout_sec=20)

    robot.logger.info("Commanding robot to stand...")
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec=10)
    robot.logger.info("Robot standing.")

    #walk_rt_vision = [-0.6707368427993169, -0.6613703096820227, 0.2523603994192693]
    # Walk to the object.
    if type(pose) == type(None):
        vision_tform_pose = stupid()
    else:
        vision_tform_pose= stupid(*pose)

    walk_rt_vision, heading_rt_vision = compute_stand_location_and_yaw(
        vision_tform_pose, robot_state_client, distance_margin=distance_margin, robot=robot)


    move_cmd = RobotCommandBuilder.trajectory_command(
        goal_x=walk_rt_vision[0],
        goal_y=walk_rt_vision[1],
        goal_heading=heading_rt_vision,
        frame_name=VISION_FRAME_NAME,
        params=get_walking_params(0.5, 0.5))

    cmd_id = command_client.robot_command(command=move_cmd,
        end_time_secs=time.time() +
        end_time)

    # Wait until the robot reports that it is at the goal.
    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=10.0, verbose=True)

def compute_stand_location_and_yaw(vision_tform_target, robot_state_client,
                                   distance_margin,robot):
    # Compute drop-off location:
    #   Draw a line from Spot to the person
    #   Back up 2.0 meters on that line
    '''
    vision_tform_robot = get_a_tform_b(
        robot_state_client.get_robot_state(
        ).kinematic_state.transforms_snapshot, VISION_FRAME_NAME,
        GRAV_ALIGNED_BODY_FRAME_NAME)
    '''

    frame_tree_snapshot = robot.get_frame_tree_snapshot()
    vision_tform_robot = get_a_tform_b(frame_tree_snapshot,"vision","hand")

    # Compute vector between robot and person
    robot_rt_person_ewrt_vision = [
        vision_tform_robot.x - vision_tform_target.x,
        vision_tform_robot.y - vision_tform_target.y,
        vision_tform_robot.z - vision_tform_target.z
    ]

    # Compute the unit vector.
    if np.linalg.norm(robot_rt_person_ewrt_vision) < 0.01:
        robot_rt_person_ewrt_vision_hat = vision_tform_robot.transform_point(1, 0, 0)
    else:
        robot_rt_person_ewrt_vision_hat = robot_rt_person_ewrt_vision / np.linalg.norm(
            robot_rt_person_ewrt_vision)

    # Starting at the person, back up meters along the unit vector.
    drop_position_rt_vision = [
        vision_tform_target.x +
        robot_rt_person_ewrt_vision_hat[0] * distance_margin,
        vision_tform_target.y +
        robot_rt_person_ewrt_vision_hat[1] * distance_margin,
        vision_tform_target.z +
        robot_rt_person_ewrt_vision_hat[2] * distance_margin
    ]

    # We also want to compute a rotation (yaw) so that we will face the person when dropping.
    # We'll do this by computing a rotation matrix with X along
    #   -robot_rt_person_ewrt_vision_hat (pointing from the robot to the person) and Z straight up:
    xhat = -robot_rt_person_ewrt_vision_hat
    zhat = [0.0, 0.0, 1.0]
    yhat = np.cross(zhat, xhat)
    mat = np.matrix([xhat, yhat, zhat]).transpose()
    heading_rt_vision = math_helpers.Quat.from_matrix(mat).to_yaw()

    return drop_position_rt_vision, heading_rt_vision

def get_walking_params(max_linear_vel, max_rotation_vel):
    max_vel_linear = geometry_pb2.Vec2(x=max_linear_vel, y=max_linear_vel)
    max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear,
                                           angular=max_rotation_vel)
    vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
    params = RobotCommandBuilder.mobility_params()
    params.vel_limit.CopyFrom(vel_limit)
    return params


def block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=None, verbose=False):
    """Helper that blocks until a trajectory command reaches STATUS_AT_GOAL or a timeout is
        exceeded.
       Args:
        command_client: robot command client, used to request feedback
        cmd_id: command ID returned by the robot when the trajectory command was sent
        timeout_sec: optional number of seconds after which we'll return no matter what the
                        robot's state is.
        verbose: if we should print state at 10 Hz.
       Return values:
        True if reaches STATUS_AT_GOAL, False otherwise.
    """
    start_time = time.time()

    if timeout_sec is not None:
        end_time = start_time + timeout_sec
        now = time.time()

    while timeout_sec is None or now < end_time:
        feedback_resp = command_client.robot_command_feedback(cmd_id)

        current_state = feedback_resp.feedback.mobility_feedback.se2_trajectory_feedback.status

        if verbose:
            current_state_str = basic_command_pb2.SE2TrajectoryCommand.Feedback.Status.Name(current_state)

            current_time = time.time()
            print('Walking: ({time:.1f} sec): {state}'.format(
                time=current_time - start_time, state=current_state_str),
                  end='                \r')

        if current_state == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL:
            return True

        time.sleep(0.1)
        now = time.time()

    if verbose:
        print('block_for_trajectory_cmd: timeout exceeded.')

    return False


if __name__ == '__main__':
    sdk = bosdyn.client.create_standard_sdk('NLMapSpot')
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)

    # Time sync is necessary so that time-based filter requests can be converted
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        move_to(robot,robot_state_client)