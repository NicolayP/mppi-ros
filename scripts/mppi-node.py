#!/usr/bin/env python3

from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg

import numpy as np
import rospy

import time as t
import sys

sys.path.append('/home/pierre/workspace/uuv_ws/\
                 src/mppi-ros/scripts/mppi_tf/scripts/')

from controller_base import ControllerBase
from cost import getCost
from model import getModel


class MPPINode(object):
    def __init__(self):
        self._state = np.zeros(13)
        self._forces = np.zeros(6)

        self._applied = []
        self._states = []
        self._accs = []

        self._initOdom = False

        self._namespace = "rexrov2"

        self._once = True

        self._elapsed = 0.
        self._steps = 0

        if rospy.has_param("~model_name"):
            self._uuvName = rospy.get_param("~model_name")
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
            self._stateDim = rospy.get_param("~state_dim")
        else:
            rospy.logerr("Don't know the state dimensionality.")
            return

        if rospy.has_param("~action_dim"):
            self._actionDim = rospy.get_param("~action_dim")
        else:
            rospy.logerr("Don't know the actuator dimensionality.")
            return

        if rospy.has_param("~cost"):
            self._task = rospy.get_param("~cost")
        else:
            rospy.logerr("No cost function given.")
            return

        if rospy.has_param("~model"):
            self._modelConf = rospy.get_param("~model")
        else:
            rospy.logerr("No internal model given.")
            return

        if rospy.has_param("~log"):
            self._log = rospy.get_param("~log")
            if rospy.has_param("~log_path"):
                self._logPath = rospy.get_param("~log_path")
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

        self._cost = getCost(self._task,
                             self._lambda,
                             self._gamma,
                             self._upsilon,
                             self._noise)

        rospy.loginfo("Get Model")

        self._model = getModel(self._modelConf,
                               self._samples,
                               self._dt,
                               True,
                               self._actionDim,
                               self._modelConf['type'])

        rospy.loginfo("Get controller")

        self._controller = ControllerBase(self._model,
                                          self._cost,
                                          k=self._samples,
                                          tau=self._horizon,
                                          dt=self._dt,
                                          s_dim=self._stateDim,
                                          a_dim=self._actionDim,
                                          lam=self._lambda,
                                          upsilon=self._upsilon,
                                          sigma=self._noise,
                                          normalize_cost=True,
                                          filter_seq=False,
                                          log=self._log,
                                          log_path=self._logPath,
                                          gif=False,
                                          debug=self._dev,
                                          config_file=None,
                                          task_file=self._task)

        rospy.loginfo("Subscrive to odometrie topics")

        # Subscribe to odometry topic
        self._odomTopicSub = rospy.Subscriber("/{}/pose_gt".
                                              format(self._uuvName),
                                              numpy_msg(Odometry),
                                              self.odometry_callback)

        rospy.loginfo("Publish to thruster topics")

        # Publish on to the thruster alocation matrix.
        self._thrustPub = rospy.Publisher(
                'thruster_input', WrenchStamped, queue_size=1)

        rospy.loginfo("Controller loaded.")

    def publish_control_wrench(self, forces):
        if not self._initOdom:
            return

        forceMsg = WrenchStamped()
        forceMsg.header.stamp = rospy.Time.now()
        forceMsg.header.frame_id = '{}/{}'.format(self._namespace, 'base_link')
        # Force
        forceMsg.wrench.force.x = forces[0]
        forceMsg.wrench.force.y = forces[1]
        forceMsg.wrench.force.z = forces[2]
        # Torque
        forceMsg.wrench.torque.x = forces[3]
        forceMsg.wrench.torque.y = forces[4]
        forceMsg.wrench.torque.z = forces[5]

        self._thrustPub.publish(forceMsg)

    def call_controller(self, state):
        start = t.perf_counter()

        self._forces = self._controller.next(state)

        end = t.perf_counter()
        self._elapsed += (end-start)
        self._steps += 1

        self.publish_control_wrench(self._forces)

    def odometry_callback(self, msg):
        # If first call, we need to boot the controller.
        if not self._initOdom:
            # First call
            self._prevTime = rospy.get_rostime()
            self._prevState = self.update_odometry(msg)
            self._initOdom = True
            self._initalState = self._prevState.copy()

        else:
            time = rospy.get_rostime()
            dt = time - self._prevTime

            if dt.to_sec() < self._dt:
                return
            self._prevTime = time
            self._state = self.update_odometry(msg)
            # save the transition
            self._controller.save(self._prevState,
                                  np.expand_dims(self._forces, -1),
                                  self._state)
            # update previous state
            self._prevState = self._state

        self.call_controller(self._prevState)

        self._applied.append(np.expand_dims(self._forces.copy(), axis=0))
        self._states.append(np.expand_dims(self._state.copy(), axis=0))

        if self._steps % 200 == 0:
            self.log()

    def log(self):
        path = "/home/pierre/workspace/uuv_ws/src/mppi-ros/log/"
        self._controller.save_rp("{}transitons.npz".format(path))
        with open("{}applied.npy".format(path), "wb") as f:
            np.save(f, np.concatenate(self._applied, axis=0))
        with open("{}init_state.npy".format(path), "wb") as f:
            np.save(f, self._initalState)
        with open("{}states.npy".format(path), "wb") as f:
            np.save(f, np.concatenate(self._states, axis=0))
        rospy.loginfo("Saved applied actions and inital state to file")

    def update_odometry(self, msg):
        """Odometry topic subscriber callback function."""
        # The frames of reference delivered by the odometry seems to be as
        # follows
        # position -> world frame
        # orientation -> world frame
        # linear velocity -> world frame
        # angular velocity -> world frame

        if self._model._inertialFrameId != msg.header.frame_id:
            raise rospy.ROSException('The inertial frame ID used by the '
                                     'vehicle model does not match the '
                                     'odometry frame ID, vehicle=%s, odom=%s' %
                                     (self._model._inertialFrameId,
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
        linVel = np.array([msg.twist.twist.linear.x,
                           msg.twist.twist.linear.y,
                           msg.twist.twist.linear.z])
        # Transform linear velocity to the BODY frame
        rotItoB = self._model.rotBtoI_np(state[3:7, 0]).T

        linVel = np.expand_dims(np.dot(rotItoB, linVel), axis=-1)
        # Angular velocity in the INERTIAL frame
        angVel = np.array([msg.twist.twist.angular.x,
                           msg.twist.twist.angular.y,
                           msg.twist.twist.angular.z])
        # Transform angular velocity to BODY frame
        angVel = np.expand_dims(np.dot(rotItoB, angVel), axis=-1)
        # Store velocity vector
        state[7:13, :] = np.concatenate([linVel, angVel], axis=0)
        return state


if __name__ == "__main__":
    print("Mppi - DP Controller")
    rospy.init_node("MPPI_DP_CONTROLLER")

    try:
        node = MPPINode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")
