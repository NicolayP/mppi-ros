#!/usr/bin/env python3
import rospy
from mppi_ros.msg import Transition, NNLayer, Tensor2d, Tensor1d
from mppi_ros.srv import UpdateModelParam, UpdateModelParamResponse
from rospy.numpy_msg import numpy_msg
import tensorflow as tf

# Hide GPU from visible devices as this 
# will run on the cpu.
tf.config.set_visible_devices([], 'GPU')

from mppi_tf.scripts.learner import get_learner
from mppi_tf.scripts.model import get_model


class MPPILearnerNode(object):
    def __init__(self):
        rospy.loginfo("Learner for MPPI...")

        if rospy.has_param("~model"):
            modelConf = rospy.get_param("~model")
        else:
            rospy.logerr("No internal model given.")
            return

        rospy.loginfo("Get Model.")


        self._model = get_model(modelConf,
                                1,
                                1,
                                True,
                                1,
                                "auv_nn")

        rospy.loginfo("Get Learner.")
        self._learner = get_learner(self._model)

        rospy.loginfo("Set Subscribers.")
        self._transTopicSub = rospy.Subscriber("/mppi/transition",
                                               numpy_msg(Transition),
                                               self.save_transition)

        rospy.loginfo("Set Services.")
        self._updateService = rospy.Service("mppi/update_model_params",
                                            UpdateModelParam,
                                            self.update_weights)

        rospy.loginfo("Learner loaded.")

    def update_weights(self, req):
        rospy.loginfo(req.train)
        if req.train:
            rospy.loginfo("Training")
            self._learner.train_epoch()
            rospy.loginfo("Training done")
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
        retmsg = UpdateModelParamResponse(layers)
        return retmsg

    def save_transition(self, msg):
        x = msg.x
        u = msg.u
        xNext = msg.xNext
        self._learner.add_rb(x, u, xNext)

if __name__ == "__main__":
    rospy.loginfo("Mppi - Learner")
    rospy.init_node("MPPI_LEARNER")

    try:
        node = MPPILearnerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Caught exception")
    print("Exiting")