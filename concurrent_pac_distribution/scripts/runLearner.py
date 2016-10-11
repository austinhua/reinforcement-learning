#!/usr/bin/env python

__author__ = 'Jason Pazis'

import rospy
from std_msgs.msg import String
from numpy import *
from time import time
from threading import Lock
from Distance import Distance
from ConcurrentPAC import ConcurrentPAC
from concurrent_pac.msg import Sample
from concurrent_pac.msg import TildeU
from concurrent_pac.srv import EnvironmentDescription
from concurrent_pac.srv import TildeUservice
from concurrent_pac.srv import TildeUserviceResponse

class Learner(object):
    def __init__(self):
        rospy.init_node('learner', anonymous=False)
        dKnown = rospy.get_param('dKnown')
        epsilonA = rospy.get_param('epsilonA')
        epsilonB = rospy.get_param('epsilonB')
        normOrder = float(rospy.get_param('normOrder'))
        #regularizer = [1.0, 1.0, 10.0, 150.0]
        regularizer = [1.0/1000.0]

        print("dKnown", dKnown, "epsilonA", epsilonA, "epsilonB", epsilonB,
              "normOrder", normOrder, "regularizer", regularizer)
        self.distance = Distance(dKnown=dKnown, normOrder=normOrder, regularizer=array(regularizer))
        self.mutex = Lock()

        # Wait until the domain is up to provide its description
        rospy.wait_for_service('EnvironmentDescription')
        environmentDescription = rospy.ServiceProxy('EnvironmentDescription', EnvironmentDescription)
        self.domain = environmentDescription()
        print("domain", self.domain)
        self.learner = ConcurrentPAC(Qmax=self.domain.Qmax, distance=self.distance, gamma=self.domain.gamma,
                                     Ka=[1, 2, 4], epsilonA=epsilonA, epsilonB=epsilonB, stateLength=self.domain.stateLength,
                                     actionLength=self.domain.actionLength, MDPlength=self.domain.MDPLength,
                                     mutex=self.mutex)
        self.UstrideLength = self.domain.stateLength+self.domain.actionLength+self.domain.MDPLength

        # Subscribe to samples from domains
        rospy.Subscriber("Sample", Sample, self.SampleCallback)

        # Start the 'TildeU' service, that will handle TildeU requests from policies
        self.actionService = rospy.Service('TildeU', TildeUservice, self.TildeUserviceHandler)

        # Start TildeU publisher, which publishes a new TildeU every time the policy changes
        self.tildeUpub = rospy.Publisher('TildeU', TildeU, queue_size=1)
        while not rospy.is_shutdown():
            if self.learner.step():
                self.mutex.acquire()
                tildeU = TildeU()
                tildeU.Qmax = self.domain.Qmax
                tildeU.dKnown = self.distance.dKnown
                tildeU.normOrder = self.distance.normOrder
                tildeU.regularizer = self.distance.regularizer
                tildeU.UstrideLength = self.UstrideLength
                tildeU.regularizedU = [item for sublist in self.learner.policy.regularizedU.copy() for item in sublist]
                tildeU.EpsilonBK = self.learner.policy.EpsilonBK.copy().tolist()
                tildeU.F = self.learner.policy.F.copy().tolist()
                self.mutex.release()
                self.tildeUpub.publish(tildeU)

    def TildeUserviceHandler(self, req):
        self.mutex.acquire()
        response = TildeUserviceResponse(self.domain.Qmax, self.distance.dKnown, self.distance.normOrder,
                                         self.distance.regularizer, self.UstrideLength,
                                  [item for sublist in self.learner.policy.regularizedU.copy() for item in sublist],
                                  self.learner.policy.EpsilonBK.copy().tolist(), self.learner.policy.F.copy().tolist())
        self.mutex.release()
        return response

    def SampleCallback(self, data):
        AsLen = len(data.As)
        actionLen = len(data.action)
        As = []
        for i in range(AsLen/actionLen):
            a = []
            for j in range(actionLen):
                index = int(i*actionLen+j)
                a.append(data.As[index])
            As.append(a)
        self.learner.addSampleToTheCandidateQueue(data.state, data.action, data.reward, data.nextState, As,
                                                  data.absorbing, data.MDP)

if __name__ == '__main__':
    try:
        learner = Learner()
    except rospy.ROSInterruptException:
        pass
