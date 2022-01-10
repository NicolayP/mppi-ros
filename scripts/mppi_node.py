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

from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from mppi_ros.msg import Transition
from mppi_ros.srv import UpdateModelParam, SaveRb, WriteWeights, WriteWeightsResponse, SetLogPath

import numpy as np
import time as t
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


class MPPINode(object):
    def __init__(self):
        self._state = np.zeros((13, 1))
        self._forces = np.zeros(6)

        self._applied = []
        self._states = []
        self._accs = []

        self._initOdom = False

        self._namespace = "rexrov2"

        self._once = True

        self._elapsed = 0.
        self._steps = 0
        self._timeSteps = 0.

        self.load_ros_params()

        if "learnable" in self._modelConf:
            self._learnable = self._modelConf["learnable"]
        else:
            self._learnable = False


        if self._log:
            stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
            path = 'graphs/python/'
            if self._dev:
                path = os.path.join(path, 'debug')
            self._logPath = os.path.join(self._logPath,
                                         path,
                                         self._modelConf['type'],
                                         "k" + str(self._samples.numpy()),
                                         "T" + str(self._horizon),
                                         "L" + str(self._lambda),
                                         stamp)
            if self._learnable:
                self.set_learner_path()

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
                                          dt=self._dt,
                                          sDim=self._stateDim,
                                          aDim=self._actionDim,
                                          lam=self._lambda,
                                          upsilon=self._upsilon,
                                          sigma=self._noise,
                                          normalizeCost=True,
                                          filterSeq=False,
                                          log=self._log,
                                          logPath=self._logPath,
                                          gif=False,
                                          graphMode=self._graphMode,
                                          debug=self._dev,
                                          configDict=None,
                                          taskDict=self._task,
                                          modelDict=self._modelConf)

        rospy.loginfo("Subscrive to odometrie topics...")

        # Subscribe to odometry topic
        self._odomTopicSub = rospy.Subscriber("/{}/pose_gt".
                                              format(self._uuvName),
                                              numpy_msg(Odometry),
                                              self.odometry_callback)
        rospy.loginfo("Done")

        rospy.loginfo("Setup publisher to thruster topics...")

        # Publish on to the thruster alocation matrix.
        self._thrustPub = rospy.Publisher(
                'thruster_input', WrenchStamped, queue_size=1)

        self._transPub = rospy.Publisher(
                '/mppi/controller/transition',
                Transition,
                queue_size=1
        )
        rospy.loginfo("Done")
        
        if self._learnable:
            rospy.loginfo("Creating service for parameter update...")
            self._writeWeightsSrv = rospy.Service("mppi/controller/write_weights",
                                                  WriteWeights,
                                                  self.write_weights)

        rospy.loginfo("Trace the tensorflow computational graph...")
        # TODO: run the controller "a blanc" to generate the tensroflow
        # graph one before starting to run it.
        start = t.perf_counter()
        if self._graphMode:
            self._controller.trace()

        end = t.perf_counter()
        rospy.loginfo("Tracing done in {:.4f} s".format(end-start))

        #rospy.loginfo("Proifile the controller...")
        #self._controller.profile()

        # reset controller.
        rospy.loginfo("Controller loaded.")

    def load_ros_params(self):
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

    def publish_transition(self, x, u, xNext):

        # Header
        transMsg = Transition()
        transMsg.header.stamp = rospy.Time.now()
        transMsg.header.frame_id = '{}/{}'.format(self._namespace, 'base_link')

        # Transition
        transMsg.x = x
        transMsg.u = u
        transMsg.xNext = xNext

        self._transPub.publish(transMsg)

    def call_controller(self, state):
        start = t.perf_counter()

        self._forces = self._controller.next(state)

        end = t.perf_counter()
        self._elapsed += (end-start)
        self._timeSteps += 1
        self._steps += 1
        self.publish_control_wrench(self._forces)

        if self._steps % 10 == 0:
            rospy.loginfo("*"*5 + " MPPI Time stats " + "*"*5)
            rospy.loginfo("* Next step : {:.4f} (sec)".format(self._elapsed/self._timeSteps))
            self._elapsed = 0.
            self._timeSteps = 0

    def odometry_callback(self, msg):
        # If first call, we need to boot the controller.
        if not self._initOdom:
            # First call
            self._prevTime = rospy.get_rostime()
            self._prevState = self.update_odometry(msg)
            self._state = self._prevState.copy()
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

            # Send the data for the learner.
            self.publish_transition(self._prevState,
                                    np.expand_dims(self._forces, -1),
                                    self._state)

            # update previous state
            self._prevState = self._state

        self.call_controller(self._prevState)

        self._applied.append(np.expand_dims(self._forces.copy(), axis=0))
        self._states.append(np.expand_dims(self._state.copy(), axis=0))

        if self._steps % 200 == 0:
            # should place that in a separte loop.
            self.log()
        
        if self._steps % 50 == 0:
            if self._learnable:
                self.update_model()

    def log(self):
        path = "/home/pierre/workspace/uuv_ws/src/mppi_ros/log/"
        self.save_rb("{}transitons.npz".format(path))
        with open("{}applied.npy".format(path), "wb") as f:
            np.save(f, np.concatenate(self._applied, axis=0))
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

    def write_weights(self, req):
        self._model.set_var(req.weights)
        return WriteWeightsResponse(True)

    def update_model(self):
        rospy.loginfo("Updating parameter model")
        rospy.wait_for_service('/mppi/learner/update_model_params')
        try:
            start = t.perf_counter()
            updateModelSrv = rospy.ServiceProxy('/mppi/learner/update_model_params', UpdateModelParam)
            resp = updateModelSrv(train=True, save=self._log, step=self._steps)
            end = t.perf_counter()
            rospy.loginfo("Service replied in {:.4f} s".format(end-start))
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def save_rb(self, file):
        rospy.loginfo("Creating client for save service..")
        rospy.wait_for_service('/mppi/learner/save_rb')
        try:
            saveRbSrv = rospy.ServiceProxy('/mppi/learner/save_rb', SaveRb)
            resp = saveRbSrv(file)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        rospy.loginfo("Done")

    def set_learner_path(self):
        rospy.loginfo("Creating client for update service..")
        rospy.wait_for_service('/mppi/learner/set_log_path')
        try:
            setLogPath = rospy.ServiceProxy('/mppi/learner/set_log_path', SetLogPath)
            resp = setLogPath(self._logPath)
            print(resp)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        rospy.loginfo("Done")


if __name__ == "__main__":
    print("Mppi - DP Controller")
    rospy.init_node("MPPI_DP_CONTROLLER")

    try:
        node = MPPINode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")
