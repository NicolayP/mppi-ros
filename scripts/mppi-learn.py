#/usr/bin/env python3
import rospy
from mppi_ros.msg import Transition
from mppi_ros.srv import updateModelParam, updateModelParamResponse
from rospy.numpy_msg import numpy_msg

class MPPILearnerNode(object):
    def __init__(self, model):
        rospy.loginfo("Learner for MPPI...")

        self._model = model
        self._learner = get_learner()
        self._updateService

        self._transTopicSub = rospy.Subscriber("/mppi/transition",
                                               numpy_msg(Transition),
                                               self.save_transition)

        self._updateService = rospy.Service("mppi/update_model_params",
                                            updateModelParam,
                                            self.update_weights)

    def update_weights(self):
        pass

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