#!/usr/bin/env python

__author__ = 'Jason Pazis'

import rospy
from numpy import *
from time import time
from threading import Lock
from Distance import Distance
from Policy import Policy
from concurrent_pac.msg import Sample
from concurrent_pac.msg import TildeU
from concurrent_pac.srv import TildeUservice
from concurrent_pac.srv import EnvironmentDescription
from concurrent_pac.srv import Action
from concurrent_pac.srv import ActionResponse


class RunPolicy(object):
    def __init__(self):
        # Allow only a single instance.
        # This one instance will service all domain action requests.
        rospy.init_node('policy', anonymous=False)

        self.mutex = Lock()     # This mutex is used to protect from using the policy while it's being updated

        # Wait for the TildeU service from the learner to become available
        rospy.wait_for_service('TildeU')
        tildeUservice = rospy.ServiceProxy('TildeU', TildeUservice)
        self.updatePolicy(tildeUservice())      # Request a TildeU from the learner and initialize TildeU

        # Subscribe to policy updates from the learner
        rospy.Subscriber("TildeU", TildeU, self.TildeUcallback)

        # Start the 'Action' service, that will handle action requests from domains
        self.actionService = rospy.Service('Action', Action, self.actionHandler)

        rospy.spin()        # keep python from exiting until this node is stopped

    def actionHandler(self, req):
        # Un-flatten As
        AsLen = len(req.As)
        As = []
        for i in range(AsLen/req.AsStrideLength):
            a = []
            for j in range(req.AsStrideLength):
                index = int(i*req.AsStrideLength+j)
                a.append(req.As[index])
            As.append(a)

        # Get an action from the internal policy
        self.mutex.acquire()
        action = self.policy.policy(req.state, req.MDP, As)
        self.mutex.release()
        return ActionResponse(action)

    def TildeUcallback(self, data):
        self.updatePolicy(data)

    def updatePolicy(self, data):
        # Un-flatten U
        Ulen = len(data.regularizedU)
        U = []
        for i in range(Ulen/data.UstrideLength):
            u = []
            for j in range(data.UstrideLength):
                index = int(i*data.UstrideLength+j)
                u.append(data.regularizedU[index])
            U.append(u)
        self.mutex.acquire()
        self.distance = Distance(dKnown=data.dKnown, normOrder=data.normOrder, regularizer=array(data.regularizer))
        self.policy = Policy(data.Qmax, self.distance, array(U), array(data.EpsilonBK), data.F)
        self.mutex.release()

if __name__ == '__main__':
    try:
        runPolicy = RunPolicy()
    except rospy.ROSInterruptException:
        pass
