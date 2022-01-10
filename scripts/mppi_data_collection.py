#!/usr/bin/env python3
from re import L
import rospy

from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

from mppi_ros.msg import Transition
from mppi_ros.srv import SaveRb

import quaternion
import numpy as np
from cpprb import ReplayBuffer
from tqdm import tqdm
import time


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
    def __init__(self, sDim, aDim):
        self._forces = np.zeros(6)
        self.load_ros_params()
        # create observer/learner.
        self.rb = ReplayBuffer(self._bufferSize,
                               env_dict={"obs": {"shape": (sDim, 1)},
                               "act": {"shape": (aDim, 1)},
                               "next_obs": {"shape": (sDim, 1)}
                               }
                              )
        # publisher to thrusters.
        self._thrustPub = rospy.Publisher(
            'thruster_input', WrenchStamped, queue_size=1)

        # subscriber to pose_gt
        self._odomSub = rospy.Subscriber("{}/pose_gt".
                                            format(self._uuvName),
                                         numpy_msg(Odometry),
                                         self.odom_callback)
        self._run = False
        self.collect_data(self._n, self._logFile)
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

        if rospy.has_param("~log_file"):
            self._logFile = rospy.get_param("~log_file")
        else:
            rospy.logerr("Need to give a saveing file.")
        
        if rospy.has_param("~buffer_size"):
            self._bufferSize = rospy.get_param("~buffer_size")
        else:
            self._bufferSize = self._n*self._maxSteps

        if rospy.has_param("~uuv_name"):
            self._uuvName = rospy.get_param("~uuv_name")
        else:
            rospy.logerr("Need to specify the vehicule's name")

        if rospy.has_param("~dt"):
            self._dt = rospy.get_param("~dt")
        else:
            rospy.logerr("Did not sepcify the delta t.")

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
        pDot = np.random.rand(3)*5
        rDot = np.random.rand(3)*0.5
        state.model_name = self._uuvName
        state.pose.position.x = p[0]
        state.pose.position.y = p[1]
        state.pose.position.z = p[2]-10

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

    def collect_data(self, n, filename):
        # launch self.run n times
        # Save observer transitions to file.
        rospy.loginfo("Start recording")
        for i in tqdm(range(n)):
            if self._run == False:
                self.run()
            while self._run:
                time.sleep(1)
        rospy.loginfo("Stop recording")
        self.save_rb(filename)

    def odom_callback(self, msg):
        time = rospy.get_rostime()
        if not self._run or self._first:
            self._prevTime = time
            self._prevState = self.update_odometry(msg)
            if self._first and self._run:
                self._first = False
            return
        
        dt = time - self._prevTime
        if dt.to_sec() < self._dt:
            return
        self._prevTime = time
        self._state = self.update_odometry(msg)

        self.save_transition(self._prevState,
                             np.expand_dims(self._forces, -1),
                             self._state)
        
        self._forces = np.random.rand(6)*100
        self.publish_control_wrench(self._forces.copy())
        
        self._prevState = self._state.copy()
        
        self._step += 1
        if self._step >= self._maxSteps:
            self._run = False
            self.stop()

    def update_odometry(self, msg):
        
        """Odometry topic subscriber callback function."""
        # The frames of reference delivered by the odometry seems to be as
        # follows
        # position -> world frame
        # orientation -> world frame
        # linear velocity -> world frame
        # angular velocity -> world frame

        # Update the velocity vector
        # Update the pose in the inertial frame
        state = np.zeros((13, 1))
        state[0:3, :] = np.array([[msg.pose.pose.position.x],
                                  [msg.pose.pose.position.y],
                                  [msg.pose.pose.position.z]])

        # Using the (w, x, y, z) format for quaternions
        state[3:7, :] = np.array([[msg.pose.pose.orientation.x],
                                  [msg.pose.pose.orientation.y],
                                  [msg.pose.pose.orientation.z],
                                  [msg.pose.pose.orientation.w]])

        # Linear velocity on the INERTIAL frame
        linVel = np.array([msg.twist.twist.linear.x,
                           msg.twist.twist.linear.y,
                           msg.twist.twist.linear.z])

        # Transform linear velocity to the BODY frame
        # TODO: Change this to a quaternion rotation. AVOID rotation matrix
        rotItoB = rotBtoI_np(state[3:7, 0]).T

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

    def save_transition(self, x, u, xNext):
        self.rb.add(obs=x, act=u, next_obs=xNext)

    def save_rb(self, filename):
        rospy.loginfo("Save transtions")
        trans = self.rb.get_all_transitions()

        print("X: ", trans['obs'].shape)
        print("U: ", trans['act'].shape)
        print("Xnext: ", trans['next_obs'].shape)
        self.rb.save_transitions(filename)
        rospy.loginfo("Done")
        return

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


if __name__ == "__main__":
    print("Mppi - Data Collection")
    rospy.init_node("MPPI_DP_CONTROLLER")

    try:
        node = MPPIDataCollection(13, 6)
        # rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")
