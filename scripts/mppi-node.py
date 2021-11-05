#!/usr/bin/env python3

import sys
sys.path.append('/home/pierre/workspace/uuv_ws/src/mppi-ros/scripts/mppi_tf/scripts/')

import argparse
import os
import time as t

from controller_base import ControllerBase
from cost import getCost
from model import getModel

from geometry_msgs.msg import WrenchStamped, PoseStamped, TwistStamped, \
    Vector3, Quaternion, Pose
from std_msgs.msg import Time
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg

import numpy as np
from quaternion import as_euler_angles
import rospy


def euler_rot(rotBtoI):
    """`list`: Orientation in Euler angles in radians
    as described in Fossen, 2011.
    """
    # Rotation matrix from BODY to INERTIAL
    rot = rotBtoI
    # Roll
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    # Pitch, treating singularity cases
    den = np.sqrt(1 - rot[2, 0]**2)
    pitch = -np.arctan(rot[2, 0] / den)
    # Yaw
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw])

class MPPINode(object):
    def __init__(self):
        self.state = np.zeros(13)
        self.forces = np.zeros(6)

        self.applied = []
        self.states = []
        self.accs = []

        self._init_odom = False

        self._namespace = "rexrov2"

        self.once = True

        self.elapsed = 0.
        self.steps = 0

        if rospy.has_param("~model_name"):
            self._uuv_name = rospy.get_param("~model_name")
        else:
            rospy.logerr("Need to specify the model name to publish on")
            return

        if rospy.has_param("~samples"):
            self._samples = rospy.get_param("~samples")
        else:
            rospy.logerr("Need to set the number of samples to use")
            return

        if rospy.has_param("~horizon"):
            self._horizon = rospy.get_param("~horizon")
        else:
            rospy.logerr("Need to set the number of samples to use")
            return

        if rospy.has_param("~lambda"):
            self._lambda = rospy.get_param("~lambda")
        else:
            rospy.logerr("Need to set the number of samples to use")
            return

        if rospy.has_param("~gamma"):
            self._gamma = rospy.get_param("~gamma")
        else:
            rospy.logerr("Need to set the number of samples to use")
            return

        if rospy.has_param("~upsilon"):
            self._upsilon = rospy.get_param("~upsilon")
        else:
            rospy.logerr("Need to set the number of samples to use")
            return

        if rospy.has_param("~dt"):
            self._dt = rospy.get_param("~dt")
        else:
            rospy.logerr("Don't know the timestep.")
            return

        if rospy.has_param("~state_dim"):
            self._state_dim = rospy.get_param("~state_dim")
        else:
            rospy.logerr("Don't know the state dimensionality.")
            return

        if rospy.has_param("~action_dim"):
            self._action_dim = rospy.get_param("~action_dim")
        else:
            rospy.logerr("Don't know the actuator dimensionality.")
            return

        if rospy.has_param("~cost"):
            self.task = rospy.get_param("~cost")
        else:
            rospy.logerr("No cost function given.")
            return

        if rospy.has_param("~model"):
            self.model_conf = rospy.get_param("~model")
        else:
            rospy.logerr("No internal model given.")
            return

        if rospy.has_param("~log"):
            self._log = rospy.get_param("~log")
            if rospy.has_param("~log_path"):
                self._log_path = rospy.get_param("~log_path")
        else:
            rospy.logerr("No log flag given.")

        if rospy.has_param("~dev"):
            self._dev = rospy.get_param("~dev")
        else:
            rospy.logerr("No flag for dev mode given.")


        if rospy.has_param("~noise"):
            self._noise = rospy.get_param("~noise")
        else:
            rospy.logerr("No noise given")

        rospy.loginfo("Get cost")

        self.cost = getCost(self.task, 
                            self._lambda, self._gamma, self._upsilon, 
                            self._noise)
        
        rospy.loginfo("Get Model")

        self.model = getModel(self.model_conf, self._samples, self._dt, True, self._action_dim, self.model_conf['type'])
        
        rospy.loginfo("Get controller")

        self.controller = ControllerBase(self.model, self.cost,
                                         k=self._samples, tau=self._horizon, dt=self._dt,
                                         s_dim=self._state_dim, a_dim=self._action_dim,
                                         lam=self._lambda, upsilon=self._upsilon,
                                         sigma=self._noise, 
                                         normalize_cost=True, filter_seq=False,
                                         log=self._log, log_path=self._log_path,
                                         gif=False, debug=self._dev,
                                         config_file=None, task_file=self.task)
        
        rospy.loginfo("Subscrive to odometrie topics")

        # Subscribe to odometry topic
        self._odom_topic_sub = rospy.Subscriber(
            "/{}/pose_gt".format(self._uuv_name), numpy_msg(Odometry), self.odometry_callback)

        rospy.loginfo("Publish to thruster topics")

        # Publish on to the thruster alocation matrix.
        self._thrust_pub = rospy.Publisher(
                'thruster_input', WrenchStamped, queue_size=1)

        rospy.loginfo("Controller loaded.")

    def publish_control_wrench(self, forces):
        if not self.odom_is_init:
            return

        force_msg = WrenchStamped()
        force_msg.header.stamp = rospy.Time.now()
        force_msg.header.frame_id = '{}/{}'.format(self._namespace, 'base_link')
        # Force
        force_msg.wrench.force.x = forces[0]
        force_msg.wrench.force.y = forces[1]
        force_msg.wrench.force.z = forces[2]
        # Torque
        force_msg.wrench.torque.x = forces[3]
        force_msg.wrench.torque.y = forces[4]
        force_msg.wrench.torque.z = forces[5]

        self._thrust_pub.publish(force_msg)

    def odometry_callback(self, msg):

        if not self._init_odom:
            # First call
            self.prev_time = rospy.get_rostime()
            self.prev_state = self.update_odometry(msg)
            self.model.set_prev_vel(self.prev_state[:, 7:13])
            self._init_odom = True

            # compute first action
            start = t.perf_counter()

            self.forces = self.controller.next(self.prev_state)

            end = t.perf_counter()
            self.elapsed += (end-start)
            self.steps += 1

            # publish first control
            self.publish_control_wrench(self.forces)
            self.applied.append(np.expand_dims(self.forces.copy(), axis=0))
            self.states.append(np.expand_dims(self.prev_state.copy(), axis=0))
            self.inital_state = self.prev_state.copy()
            return
        
        time = rospy.get_rostime()
        dt = time - self.prev_time

        if dt.to_sec() < self._dt:
            return
        self.prev_time = time

        self.state = self.update_odometry(msg)

        #rospy.loginfo("State: {}".format(self.state))
        # save the transition
        self.controller.save(self.prev_state, np.expand_dims(self.forces, -1), self.state)

        # update previous state
        self.prev_state = self.state

        start = t.perf_counter()

        # compute first action
        self.forces = self.controller.next(self.prev_state)

        end = t.perf_counter()
        self.elapsed += (end-start)
        self.steps += 1

        if self.steps % 10 == 0:
            rospy.loginfo("*"*5 + " MPPI Time stats " + "*"*5)
            rospy.loginfo("* Next step   : {:.4f} (sec)".format(self.elapsed/self.steps))

        # publish first control
        self.publish_control_wrench(self.forces)
        self.applied.append(np.expand_dims(self.forces.copy(), axis=0))
        self.states.append(np.expand_dims(self.state.copy(), axis=0))

        if self.steps % 200 == 0:
            self.controller.save_rp("/home/pierre/workspace/uuv_ws/src/mppi-ros/log/transitons.npz")
            with open("/home/pierre/workspace/uuv_ws/src/mppi-ros/log/applied.npy", "wb") as f:
                np.save(f, np.concatenate(self.applied, axis=0))
            with open("/home/pierre/workspace/uuv_ws/src/mppi-ros/log/init_state.npy", "wb") as f:
                np.save(f, self.inital_state)
            with open("/home/pierre/workspace/uuv_ws/src/mppi-ros/log/states.npy", "wb") as f:
                np.save(f, np.concatenate(self.states, axis=0))
            rospy.loginfo("Saved applied actions and inital state to file")

    def save_paths_and_actions(self, paths, applied):
        with open("/home/pierre/workspace/uuv_ws/src/mppi-ros/log/traj.npy", "wb") as f:
            quats = paths[:, :, 3:7]
            paths_euler = np.zeros(shape=(paths.shape[0], paths.shape[1], 12, 1))
            for i, entry in enumerate(quats):
                for j, el in enumerate(entry):
                    q = np.quaternion(el[0], el[1], el[2], el[3])
                    euler = as_euler_angles(q)
                    paths_euler[i, j, 0:3] = paths[i, j, 0:3]
                    paths_euler[i, j, 3:6] = np.expand_dims(euler, axis=-1)
                    paths_euler[i, j, 6:12] = paths[i, j, 7:13]
            np.save(f, paths_euler)
            print(paths_euler.shape)
        with open("/home/pierre/workspace/uuv_ws/src/mppi-ros/log/applied.npy", "wb") as f:
            np.save(f, applied)
            print(applied.shape)

    def update_odometry(self, msg):
        """Odometry topic subscriber callback function."""
        # The frames of reference delivered by the odometry seems to be as
        # follows
        # position -> world frame
        # orientation -> world frame
        # linear velocity -> world frame
        # angular velocity -> world frame

        if self.model.inertial_frame_id != msg.header.frame_id:
            raise rospy.ROSException('The inertial frame ID used by the '
                                     'vehicle model does not match the '
                                     'odometry frame ID, vehicle=%s, odom=%s' %
                                     (self.model.inertial_frame_id,
                                      msg.header.frame_id))

        # Update the velocity vector
        # Update the pose in the inertial frame
        state = np.zeros((13, 1))
        state[0:3, :] = np.array([[msg.pose.pose.position.x],
                                  [msg.pose.pose.position.y],
                                  [msg.pose.pose.position.z]])

        # Using the (w, x, y, z) format for quaternions
        state[3:7, :] = np.array([[msg.pose.pose.orientation.w],
                                  [msg.pose.pose.orientation.x],
                                  [msg.pose.pose.orientation.y],
                                  [msg.pose.pose.orientation.z]])

        # Linear velocity on the INERTIAL frame
        lin_vel = np.array([msg.twist.twist.linear.x,
                            msg.twist.twist.linear.y,
                            msg.twist.twist.linear.z])
        # Transform linear velocity to the BODY frame
        rotItoB = self.model.rotBtoI_np(state[3:7, 0]).T
        euler = euler_rot(rotItoB.T)
        #tItoB = self.model.tItoB_np(euler)

        lin_vel = np.expand_dims(np.dot(rotItoB, lin_vel), axis=-1)
        # Angular velocity in the INERTIAL frame
        ang_vel = np.array([msg.twist.twist.angular.x,
                            msg.twist.twist.angular.y,
                            msg.twist.twist.angular.z])
        # Transform angular velocity to BODY frame
        ang_vel = np.expand_dims(np.dot(rotItoB, ang_vel), axis=-1)
        #ang_vel = np.expand_dims(ang_vel, axis=-1)
        # Store velocity vector
        state[7:13, :] = np.concatenate([lin_vel, ang_vel], axis=0)
        return state

    @property
    def odom_is_init(self):
        """`bool`: `True` if the first odometry message was received"""
        return self._init_odom

if __name__ == "__main__":
    print("Mppi - DP Controller")
    rospy.init_node("MPPI_DP_CONTROLLER")

    try:
        node = MPPINode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")