#!/usr/bin/env python3
import rospy
import tensorflow as tf

rospy.loginfo("Set GPU")
if rospy.has_param("~gpu_idx"):
    gpu_idx = rospy.get_param("~gpu_idx")
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > gpu_idx:
        tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
    else:
        rospy.logerr("GPU index out of range")
rospy.loginfo("Done")

# Import after setting the GPU otherwise tensorflow complains.
from mppi_tf.scripts.src.controller import get_controller
from mppi_tf.scripts.src.cost import get_cost
from mppi_tf.scripts.src.model import get_model


from geometry_msgs.msg import WrenchStamped, Twist
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from std_srvs.srv import Empty
from mppi_ros.msg import Transition
from mppi_ros.srv import UpdateModelParam, SaveRb, WriteWeights, WriteWeightsResponse, SetLogPath

from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState


import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import os

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


def tItoB_np(euler):
        r = euler[0]
        p = euler[1]
        T = np.array([[1., 0., -np.sin(p)],
                      [0., np.cos(r), np.cos(p) * np.sin(r)],
                      [0., -np.sin(r), np.cos(p) * np.cos(r)]])
        return T


def pertModel(modelParams, sigma):
    modelParams["mass"] += 5*np.random.normal(scale=sigma)
    modelParams["inertial"]["ixx"] += 5*np.random.normal(scale=sigma)
    modelParams["inertial"]["iyy"] += 5*np.random.normal(scale=sigma)
    modelParams["inertial"]["izz"] += 5*np.random.normal(scale=sigma)
    modelParams["inertial"]["ixy"] += np.random.normal(scale=sigma)
    modelParams["inertial"]["ixz"] += np.random.normal(scale=sigma)
    modelParams["inertial"]["iyz"] += np.random.normal(scale=sigma)
    modelParams["cog"] += 0.05*np.random.normal(scale=sigma, size=(3,))
    modelParams["cob"] += 0.05*np.random.normal(scale=sigma, size=(3,))
    modelParams["volume"] += 0.01*np.random.normal(scale=sigma)
    modelParams["length"] += 0.01*np.random.normal(scale=sigma)
    modelParams["height"] += 0.01*np.random.normal(scale=sigma)
    modelParams["width"] += 0.01*np.random.normal(scale=sigma)
    modelParams["Ma"] += 5*np.random.normal(scale=sigma, size=(6, 6))
    modelParams["linear_damping"] += 5*np.random.normal(scale=sigma, size=(6,))
    modelParams["quad_damping"] += 5*np.random.normal(scale=sigma, size=(6,))
    return modelParams


class ModelMissmatchExperiment(object):
    def __init__(self):
        self._run=False
        self._first=False
        self.load_ros_param()
        rospy.loginfo("Setup Controller...")
        rospy.loginfo("Get cost")

        self._cost = get_cost(self._task,
                              self._lambda,
                              self._gamma,
                              self._upsilon,
                              self._noise)

        rospy.loginfo("Get Model")

        self._model = get_model(self._modelConf,
                               self._samples,
                               self._dt,
                               True,
                               self._actionDim,
                               self._modelConf['type'])


        rospy.loginfo("Get controller")
        self._controller = get_controller(model=self._model,
                                          cost=self._cost,
                                          k=self._samples,
                                          tau=self._horizon,
                                          sDim=self._stateDim,
                                          aDim=self._actionDim,
                                          lam=self._lambda,
                                          upsilon=self._upsilon,
                                          sigma=self._noise,
                                          normalizeCost=True,
                                          filterSeq=False,
                                          log=self._log,
                                          logPath=self._logPath,
                                          graphMode=self._graphMode,
                                          debug=self._dev,
                                          configDict=None,
                                          taskDict=self._task,
                                          modelDict=self._modelConf)

        rospy.loginfo("Generate ground truth traj")
        self.init=np.array([[[0.], [0.], [0.],
                             [0.], [0.], [0.], [1.],
                             [0.], [0.], [0.],
                             [0.], [0.], [0.]]])

        self.actionSeq = np.random.rand(1, 10, 6, 1)*500
        self.gtTraj = self._model.run_model(self.init, self.actionSeq)

        rospy.loginfo("Subscrive to odometrie topics...")

        # Subscribe to odometry topic
        self._odomTopicSub = rospy.Subscriber("odom".
                                              format(self._uuvName),
                                              numpy_msg(Odometry),
                                              self.odom_callback)
        rospy.loginfo("Done")

        rospy.loginfo("Setup publisher to thruster topics...")

        # Publish on to the thruster alocation matrix.
        self._thrustPub = rospy.Publisher(
                'thruster_input', WrenchStamped, queue_size=1)

        # Publish on to the thruster alocation matrix.
        self._thrustPubTwist = rospy.Publisher(
                'thruster_input_twist', Twist, queue_size=1)

        self._transPub = rospy.Publisher(
                '/mppi/controller/transition',
                Transition,
                queue_size=1
        )
        rospy.loginfo("Controller loaded.")

        self.collect_data(self._n, self._dir)

    def load_ros_param(self):
        if rospy.has_param("~model_name"):
            self._uuvName = rospy.get_param("~model_name")
        else:
            rospy.logerr("Need to specify the model name to publish on")
            return

        if rospy.has_param("~samples"):
            self._samples = tf.Variable(rospy.get_param("~samples"))
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
                self._logPath = "."
        else:
            rospy.logerr("No log flag given.")

        if rospy.has_param("~dev"):
            self._dev = rospy.get_param("~dev")
        else:
            rospy.logerr("No flag for dev mode given.")

        if rospy.has_param("~graph_mode"):
            self._graphMode = rospy.get_param("~graph_mode")
        else:
            rospy.logerr("No flag for graph mode given.")

        if rospy.has_param("~noise"):
            self._noise = rospy.get_param("~noise")
        else:
            rospy.logerr("No noise given")

        if rospy.has_param("~rollouts"):
            self._n = rospy.get_param("~rollouts")
        else:
            self._n = 100
        
        if rospy.has_param("~max_steps"):
            self._maxSteps = rospy.get_param("~max_steps")
        else:
            self._maxSteps = 20

        if rospy.has_param("~max_thrust"):
            self._maxThrust = rospy.get_param("~max_thrust")
        else:
            rospy.logerr("Did not specify the max thrust of the vehicle")

        if rospy.has_param("~dir"):
            self._dir = rospy.get_param("~dir")
        else:
            rospy.logerr("No dir to save the experiment data")


        return        

    def run(self):
        # delete robot instance.
        # reset simulation and pause it
        # spawn new robot with random speed and postion.
        # rollout the robot with random actions.
        # log transtions in the observer.
        self.stop()
        self.reset()
        self.spawn()
        self.spawn_controller()
        self.rollout()
        return

    def stop(self):
        # Pauses the simulation.
        try:
            rospy.wait_for_service('/gazebo/pause_physics', 1)
            pauseSim = rospy.ServiceProxy('/gazebo/pause_physics',
                                          Empty)
            resp = pauseSim()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(2)
        pass

    def reset(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            resetSim = rospy.ServiceProxy('/gazebo/reset_world',
                                          Empty)
            resp = resetSim()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(2)
        self._traj = []
        self._elapsed = 0.
        self._timeSteps = 0.
        pass

    def spawn_controller(self):
        self._randModelConf = pertModel(self._modelConf.copy(), 1.)

        self._model = get_model(self._randModelConf,
                               self._samples,
                               self._dt,
                               True,
                               self._actionDim,
                               self._randModelConf['type'])


        rospy.loginfo("Get controller")
        self._controller = get_controller(model=self._model,
                                          cost=self._cost,
                                          k=self._samples,
                                          tau=self._horizon,
                                          sDim=self._stateDim,
                                          aDim=self._actionDim,
                                          lam=self._lambda,
                                          upsilon=self._upsilon,
                                          sigma=self._noise,
                                          normalizeCost=True,
                                          filterSeq=False,
                                          log=self._log,
                                          logPath=self._logPath,
                                          graphMode=self._graphMode,
                                          debug=self._dev,
                                          configDict=None,
                                          taskDict=self._task,
                                          modelDict=self._randModelConf)

    def spawn(self):
        state = ModelState()
        state.model_name = self._uuvName
        state.pose.position.x = 0.
        state.pose.position.y = 0.
        state.pose.position.z = -50.

        state.pose.orientation.x = 0.
        state.pose.orientation.y = 0.
        state.pose.orientation.z = 0.
        state.pose.orientation.w = 1.

        state.twist.linear.x = 0.
        state.twist.linear.y = 0.
        state.twist.linear.z = 0.

        state.twist.angular.x = 0.
        state.twist.angular.y = 0.
        state.twist.angular.z = 0.

        # spawn the robot using the gazebo service.
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            setModel = rospy.ServiceProxy('/gazebo/set_model_state',
                                          SetModelState)
            resp = setModel(state)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def rollout(self):
        # performs a rollout for `steps` steps
        self._step = 0
        self._first = True
        self._steps = 0
        self.start_sim()
        pass

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
        pass

    def odom_callback(self, msg):
        t = rospy.get_rostime()
        if not self._run or self._first:
            self._prevTime = t
            self._prevState = self.update_odometry(msg)
            if self._first and self._run:
                self._first = False
            return

        self._state = self.update_odometry(msg)

        self._traj.append(np.expand_dims(self._state.copy(), axis=0))

        self.call_controller(self._state)

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

        #if self._model._inertialFrameId != msg.header.frame_id:
        #    raise rospy.ROSException('The inertial frame ID used by the '
        #                             'vehicle model does not match the '
        #                             'odometry frame ID, vehicle=%s, odom=%s' %
        #                             (self._model._inertialFrameId,
        #                              msg.header.frame_id))

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

    def call_controller(self, state):
        start = time.perf_counter()

        self._forces = self._controller.next(state)
        # Normalize forces vector for cmdvel
        # self._forces = self._forces / np.linalg.norm(self._forces)

        end = time.perf_counter()
        self._elapsed += (end-start)
        self._timeSteps += 1
        self._steps += 1
        self.publish_control_wrench(self._forces)

        if self._steps % 10 == 0:
            rospy.loginfo("*"*5 + " MPPI Time stats " + "*"*5)
            rospy.loginfo("* Next step : {:.4f} (sec)".format(self._elapsed/self._timeSteps))
            self._elapsed = 0.
            self._timeSteps = 0.

    def collect_data(self, n, dir):
        # launch self.run n times
        # Save observer transitions to file.

        rospy.loginfo("Start recording")
        trajFilename = "model_missmatch_traj.npy"
        errFilename = "model_missmatch_error.npy"
        trajFile = os.path.join(dir, trajFilename)
        errFile = os.path.join(dir, errFilename)
        trajs = []
        modelsErr = []
        for i in tqdm(range(n)):
            # New bag for this run
            if self._run == False:
                pertModelRun = self._model.run_model(self.init, self.actionSeq)
                modelErr = tf.linalg.norm(pertModelRun - self.gtTraj)
                self.run()
            while self._run:
                time.sleep(1)
            trajs.append(np.expand_dims(
                            np.concatenate(self._traj.copy(),axis=0),
                            axis=0))
            modelsErr.append(np.expand_dims(modelErr, axis=0))
        
        modelsErr = np.concatenate(modelsErr, axis=0)
        trajs = np.concatenate(trajs, axis=0)
        
        np.save(trajFile, trajs)
        np.save(errFile, modelsErr)
        
        rospy.loginfo("Stop recording")

if __name__ == "__main__":
    rospy.loginfo("Mppi - Model Missmatch Experiment")
    rospy.init_node("MPPI_Model_Missmatch")

    try:
        node = ModelMissmatchExperiment()
    except rospy.ROSInterruptException:
        rospy.logerr("Caught exception.")
    rospy.loginfo("Exiting")