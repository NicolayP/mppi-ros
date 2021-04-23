#!/usr/bin/env python3

import argparse
import os

from tqdm import tqdm

from mppi_tf.scripts.controller_base import ControllerBase
from mppi_tf.scripts.cost import getCost
#from model import getModel

from geometry_msgs.msg import WrenchStamped, PoseStamped, TwistStamped, \
    Vector3, Quaternion, Pose
from std_msgs.msg import Time
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg

import numpy as np
import rospy
import sys


class MPPINode(object):
    def __init__(self):

         # Subscribe to odometry topic
        #self._odom_topic_sub = rospy.Subscriber(
        #    '/bluerov2/pose_gt', numpy_msg(Odometry), self.odometry_callback)

        # Publish on to the thruster alocation matrix.
        #self._thrust_pub = rospy.Publisher(
        #        'thruster_output', WrenchStamped, queue_size=1)

        self.state = np.zeros(13)
        self.forces = np.ones(6)

        self._init_odom = False

        self._namespace = "foo"

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

        if rospy.has_param("~cost"):
            self.task = rospy.get_param("~cost")
        else:
            rospy.logerr("No cost function given")
            return

        self._noise = np.array([[1., 0.], [0., 1.]])

        self.cost = getCost(self.task, 
                            self._lambda, self._gamma, self._upsilon, 
                            self._noise, self._horizon)
        '''
        self.model = getModel(self.cost, 
                              self._s_dim, self._a_dim,
                              self._model_name)

        self.controller = ControllerBase(self.model, self.cost,
                                         k=self._samples, tau=self._horizon, dt=self._dt,
                                         s_dim=self._s_dim, a_dim=self._a_dim,
                                         lam=self._lambda, upsilon=self._upsilon,
                                         sigma=self._noise, log=self._log, gif=False, 
                                         normalize_cost=True, filter_seq=True, debug=True,
                                         config_file=self.conf, task_file=self.task)
        '''
        pass 

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
        # TODO: compute the state in body frame.
        # get the new system state

        if not self._init_odom:
            self._init_odom = True   
        
        self.state[0] = msg.pose.pose.position.x
        self.state[1] = msg.pose.pose.position.y
        self.state[2] = msg.pose.pose.position.z
        
        self.state[3] = msg.pose.pose.orientation.x
        self.state[4] = msg.pose.pose.orientation.y
        self.state[5] = msg.pose.pose.orientation.z
        self.state[6] = msg.pose.pose.orientation.w

        self.state[7] = msg.twist.twist.linear.x
        self.state[8] = msg.twist.twist.linear.y
        self.state[9] = msg.twist.twist.linear.z

        self.state[10] = msg.twist.twist.angular.x
        self.state[11] = msg.twist.twist.angular.y
        self.state[12] = msg.twist.twist.angular.z
        
        #rospy.loginfo("State: {}".format(self.state))
        # save the transition
        self.controller.save(self.prev_state, self.forces, self.state)

        # compute next action
        self.forces = self.controller.next(self.state)

        # execute the control action
        self.publish_control_wrench(self.forces)

        # update previous state
        self.prev_state = self.state

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