#!/usr/bin/env python3
import rospy
from mppi_ros.msg import Transition, NNLayer, Tensor2d, Tensor1d
from mppi_ros.srv import UpdateModelParam, UpdateModelParamResponse
from mppi_ros.srv import WriteWeights
from mppi_ros.srv import SaveRb, SaveRbResponse
from mppi_ros.srv import SetLogPath, SetLogPathResponse
from rospy.numpy_msg import numpy_msg

import tensorflow as tf
# Hide GPU from visible devices as this 
# will run on the cpu.
tf.config.set_visible_devices([], 'GPU')

from mppi_tf.scripts.src.learner import get_learner
from mppi_tf.scripts.src.model import get_model

import time as t
import os

import threading

class MPPILearnerNode(object):
    def __init__(self):
        rospy.loginfo("Learner for MPPI...")

        if rospy.has_param("~model"):
            modelConf = rospy.get_param("~model")
        else:
            rospy.logerr("No internal model given.")
            return
        
        if rospy.has_param("~log"):
            log = rospy.get_param("~log")
        else:
            log = False
        
        self._logPath = None

        rospy.loginfo("Get Model.")


        self._model = get_model(modelConf,
                                1,
                                1,
                                True,
                                1,
                                "auv_nn")

        rospy.loginfo("Get Learner.")
        self._setLogPathService = rospy.Service("/mppi/learner/set_log_path",
                                                 SetLogPath,
                                                 self.set_log_path)
        self.wait_log_path()
        self._learner = get_learner(self._model, log, self._logPath)

        rospy.loginfo("Set Subscribers.")
        self._transTopicSub = rospy.Subscriber("/mppi/controller/transition",
                                               numpy_msg(Transition),
                                               self.save_transition)

        rospy.loginfo("Set Services.")
        self._updateService = rospy.Service("/mppi/learner/update_model_params",
                                            UpdateModelParam,
                                            self.update_weights)

        self._saveRbService = rospy.Service("/mppi/learner/save_rb",
                                            SaveRb,
                                            self.save_rb)

        rospy.loginfo("Creating client for write service..")
        rospy.wait_for_service('/mppi/controller/write_weights')
        try:
            self.writeWeights = rospy.ServiceProxy('/mppi/controller/write_weights',
                                                   WriteWeights)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        rospy.loginfo("Done")


        rospy.loginfo("Learner loaded.")

    def save_rb(self, req):
        dir = os.path.dirname(os.path.abspath(req.filename))
        if os.path.isdir(dir):
            self._learner.save_transitions(req.filename)
            rospy.loginfo("Transitions saved.")
            return SaveRbResponse(True)
        else:
            rospy.logwarning("Path to filename doesn't exist")
            return SaveRbResponse(False)

    def update_weights(self, req):
        x = threading.Thread(target=self.update_weights_thread, args=(req,))
        x.start()
        return UpdateModelParamResponse(True)

    def train(self):
        start = t.perf_counter()
        self._learner.train_epoch()
        end = t.perf_counter()
        rospy.loginfo("Learning done in {:.4f} s".format(end-start))

    def update_weights_thread(self, req):
        if req.train:
            self.train()
        if req.save:
            self._learner.save_params(req.step)
        weights = self.format_weights()
        rospy.loginfo("Send weights")
        rospy.wait_for_service('/mppi/learner/write_weights')
        try:
            resp = self.writeWeights(weights)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        return resp

    def format_weights(self):
        layers = []

        w = True
        for el in self._model.get_var():
            if w:
                layer = NNLayer()
                weights = Tensor2d()
                for entry in el.numpy():
                    weights.tensor.append(Tensor1d(entry))
                layer.weights_name = el.name
                layer.weights = weights
                w = False
            else:
                bias = Tensor1d()
                bias.tensor = el.numpy()
                layer.bias_name = el.name
                layer.bias = bias
                w = True
                layers.append(layer)
        
        return layers

    def save_transition(self, msg):
        x = msg.x
        u = msg.u
        xNext = msg.xNext
        self._learner.add_rb(x, u, xNext)

    def wait_log_path(self):
        rate = rospy.Rate(5)
        while self._logPath == None and ( not rospy.is_shutdown() ):
            rate.sleep()
        return

    def set_log_path(self, req):
        rospy.loginfo("Learner save_path: {}".format(req.path))
        self._logPath = req.path
        return SetLogPathResponse(True)

if __name__ == "__main__":
    rospy.loginfo("Mppi - Learner")
    rospy.init_node("MPPI_LEARNER")

    try:
        node = MPPILearnerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")