#!/usr/bin/env python3
import rospy

from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState


import quaternion
import numpy as np
from tqdm import tqdm
import time

import rosbag
import os
import random


def rotBtoI_np(quat):
    
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    return np.array([
                        [1 - 2 * (y**2 + z**2),
                        2 * (x * y - z * w),
                        2 * (x * z + y * w)],
                        [2 * (x * y + z * w),
                        1 - 2 * (x**2 + z**2),
                        2 * (y * z - x * w)],
                        [2 * (x * z - y * w),
                        2 * (y * z + x * w),
                        1 - 2 * (x**2 + y**2)]
                    ])

class MPPIDataCollection(object):
    '''
        Random data collection. Samples on the action space and apply
        it for a given duration. Define multiple strategies. Either 
        pure uniform sampling or a random walker with a specified
        stochastic process.
    '''
    def __init__(self):
        self._forces = np.zeros(6)
        self.load_ros_params()

        # publisher to thrusters.
        self._thrustPub = rospy.Publisher(
            'thruster_input', WrenchStamped, queue_size=1)

        # subscriber to pose_gt
        self._odomSub = rospy.Subscriber("{}/pose_gt".
                                            format(self._uuvName),
                                         numpy_msg(Odometry),
                                         self.odom_callback)
        self._run = False
        self.collect_data(self._n, self._logDir)
        pass

    def load_ros_params(self):
        if rospy.has_param("~rollouts"):
            self._n = rospy.get_param("~rollouts")
        else:
            self._n = 100
        
        if rospy.has_param("~max_steps"):
            self._maxSteps = rospy.get_param("~max_steps")
        else:
            self._maxSteps = 20

        if rospy.has_param("~log_dir"):
            self._logDir = rospy.get_param("~log_dir")
            if os.path.exists(self._logDir):
                rospy.loginfo("Saving directory already exists.")
            else:
                os.mkdir(self._logDir)
        else:
            rospy.logerr("Need to give a saveing directory.")

        if rospy.has_param("~uuv_name"):
            self._uuvName = rospy.get_param("~uuv_name")
        else:
            rospy.logerr("Need to specify the vehicule's name")

        if rospy.has_param("~dt"):
            self._dt = rospy.get_param("~dt")
        else:
            rospy.logerr("Did not sepcify the delta t.")

        if rospy.has_param("~max_thrust"):
            self._maxThrust = rospy.get_param("~max_thrust")
            self._std = 0.1*self._maxThrust
        else:
            rospy.logerr("Did not specify the max thrust of the vehicle")

    # Utility methods to reset the simulator.
    def run(self):
        # delete robot instance.
        # reset simulation and pause it
        # spawn new robot with random speed and postion.
        # rollout the robot with random actions.
        # log transtions in the observer.
        self.stop()
        self.reset()
        self.spawn()
        self.rollout()

    def stop(self):
        # Resets the simulation and pauses it.
        try:
            rospy.wait_for_service('/gazebo/pause_physics', 1)
            pauseSim = rospy.ServiceProxy('/gazebo/pause_physics',
                                          Empty)
            resp = pauseSim()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(2)

    def reset(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            resetSim = rospy.ServiceProxy('/gazebo/reset_world',
                                          Empty)
            resp = resetSim()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(2)

    def spawn(self):
        # generate random state dict.
        state = ModelState()
        p = np.random.rand(3)
        q = quaternion.as_quat_array(np.random.rand(4))
        pDot = np.random.rand(3)*10
        rDot = np.random.rand(3)*1.5
        state.model_name = self._uuvName
        state.pose.position.x = p[0]
        state.pose.position.y = p[1]
        state.pose.position.z = p[2]-50

        state.pose.orientation.x = q.x
        state.pose.orientation.y = q.y
        state.pose.orientation.z = q.z
        state.pose.orientation.w = q.w

        state.twist.linear.x = pDot[0]
        state.twist.linear.y = pDot[1]
        state.twist.linear.z = pDot[2]

        state.twist.angular.x = rDot[0]
        state.twist.angular.y = rDot[1]
        state.twist.angular.z = rDot[2]

        state.reference_frame = 'world'

        # spawn the robot using the gazebo service.
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            setModel = rospy.ServiceProxy('/gazebo/set_model_state',
                                          SetModelState)
            resp = setModel(state)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        
        self._prevState = np.zeros((13, 1))
        self._prevState[0:3, 0] = p
        self._prevState[3, 0] = q.x
        self._prevState[4, 0] = q.y
        self._prevState[5, 0] = q.z
        self._prevState[6, 0] = q.w
        self._prevState[7:13, 0] = np.concatenate([pDot, rDot], axis=0)

    def rollout(self):
        # performs a rollout for `steps` steps
        self._step = 0
        self._first = True
        self.start_sim()

    def start_sim(self):
        # Starts the simulation when everything is ready
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            pauseSim = rospy.ServiceProxy('/gazebo/unpause_physics',
                                          Empty)
            resp = pauseSim()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(0.25)
        t = rospy.get_rostime()
        self._prevTime = t
        self._run = True

    def collect_data(self, n, dir):
        # launch self.run n times
        # Save observer transitions to file.

        rospy.loginfo("Start recording")
        for i in tqdm(range(n)):
            # New bag for this run
            filename = "run{}.bag".format(i)
            file = os.path.join(dir, filename)
            self.bag = rosbag.Bag(file, 'w')

            if self._run == False:
                self.uniform = bool(random.randint(0, 1))
                self.uniform = True
                self.run()
            while self._run:
                time.sleep(1)

            # Close this bag
            self.bag.close()
        rospy.loginfo("Stop recording")

    def next(self):
        t = rospy.get_rostime()

        if self._dt > (t - self._prevTime).to_sec():
            return self._forces
        else:
            if self.uniform:
                forces = np.random.uniform(low=-self._maxThrust, high=self._maxThrust, size=(6))
            else:
                forces = self._forces + np.random.normal(loc=0, scale=self._std, size=(6))
            self._prevTime = t
        return forces

    def odom_callback(self, msg):
        t = rospy.get_rostime()
        if not self._run or self._first:
            self._prevTime = t
            self._prevState = self.update_odometry(msg)
            if self._first and self._run:
                self._first = False
            return

        self._state = self.update_odometry(msg)

        self._forces = self.next()
        self.publish_control_wrench(self._forces.copy())

        self._step += 1
        if self._step >= self._maxSteps:
            self._run = False
            self.stop()

    def update_odometry(self, msg):
        """Odometry topic subscriber callback function."""
        if self._run:
            self.bag.write("/{}/pose_gt".format(self._uuvName), msg)

    def publish_control_wrench(self, forces):
        forceMsg = WrenchStamped()
        forceMsg.header.stamp = rospy.Time.now()
        forceMsg.header.frame_id = '{}/{}'.format(self._uuvName,
                                                  'base_link')

        # Force
        forceMsg.wrench.force.x = forces[0]
        forceMsg.wrench.force.y = forces[1]
        forceMsg.wrench.force.z = forces[2]
        # Torque
        forceMsg.wrench.torque.x = forces[3]
        forceMsg.wrench.torque.y = forces[4]
        forceMsg.wrench.torque.z = forces[5]

        self._thrustPub.publish(forceMsg)
        self.bag.write("/thruster_input", forceMsg)


if __name__ == "__main__":
    print("Mppi - Data Collection")
    rospy.init_node("MPPI_DP_CONTROLLER")

    try:
        node = MPPIDataCollection()
        # rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")
