#!/usr/bin/env python

__author__ = 'Jason Pazis'

import rospy
from std_msgs.msg import String
from numpy import *
from copy import deepcopy
from time import sleep
from concurrent_pac.msg import Sample
from concurrent_pac.msg import EpisodeStats
from concurrent_pac.srv import EnvironmentDescription
from concurrent_pac.srv import EnvironmentDescriptionResponse
from concurrent_pac.srv import Action
from Pendulum import Pendulum
from PendulumSwingup import PendulumSwingup
from HIVtreatment import HIVtreatment

class Domain(object):
    def __init__(self):
        domain = rospy.get_param('domain')
        self.domain = globals()[domain]()
        self.stepsPerSecond = rospy.get_param('stepsPerSecond')
        self.maxStepsPerEpisode = rospy.get_param('maxStepsPerEpisode')
        self.episodeStepsRemaining = self.maxStepsPerEpisode
        self.maxNumberOfEpisodes = rospy.get_param('maxNumberOfEpisodes')
        print("domain:", domain, "stepsPerSecond:", self.stepsPerSecond, "maxStepsPerEpisode:", self.maxStepsPerEpisode,
              "episodeStepsRemaining:", self.episodeStepsRemaining, "maxNumberOfEpisodes:", self.maxNumberOfEpisodes)

        # Allow multiple instances to run simultaneously
        # (the name will contain an anonymized string)
        rospy.init_node('Domain', anonymous=True)
        self.environmentDescriptionService = rospy.Service('EnvironmentDescription', EnvironmentDescription,
                                                           self.environmentDescriptionHandler)
        self.samplePub = rospy.Publisher('Sample', Sample, queue_size=10)
        self.EpisodeStatsPub = rospy.Publisher('EpisodeStats', EpisodeStats, queue_size=100)
        self.rate = rospy.Rate(self.stepsPerSecond)
        # Make sure that the Action service is up before calling it
        rospy.wait_for_service('Action')
        self.actionService = rospy.ServiceProxy('Action', Action)
        nextState, nextMDP, nextAs = self.domain.start()
        sleep(random.uniform(1.0, 10.0))
        nextAction = array(self.actionService(nextState, nextMDP, len(self.domain.As[0]),  nextAs).action)
        accumulatedReward = 0.0
        accumulatedDiscountedReward = 0.0
        discount = 1.0
        episodeNo = 1
        while (not rospy.is_shutdown()) and (episodeNo <= self.maxNumberOfEpisodes):
            sample = self.domain.step(nextAction)       # Perform a single step in the domain

            # Package the latest sample from the domain into a message
            sampleMessage = Sample()
            sampleMessage.state = sample[0]
            sampleMessage.action = sample[1]
            sampleMessage.reward = sample[2]
            sampleMessage.nextState = sample[3]
            sampleMessage.As = sample[4]
            sampleMessage.absorbing = sample[5]
            sampleMessage.MDP = sample[6]
            self.samplePub.publish(sampleMessage)       # Publish the sample so the learner can process it
            accumulatedReward += sampleMessage.reward
            accumulatedDiscountedReward += discount*sampleMessage.reward
            discount *= self.domain.gamma

            if self.episodeStepsRemaining > 0:
                self.episodeStepsRemaining -= 1
            # If an absorbing state has been encountered, or the number of
            # steps in this episode has reached the maximum number of allowed steps,
            # reset the domain
            if sampleMessage.absorbing or self.episodeStepsRemaining == 0:
                episodeStatsMessage = EpisodeStats()
                episodeStatsMessage.episodeNo = episodeNo
                episodeStatsMessage.numberOfsteps = self.maxStepsPerEpisode-self.episodeStepsRemaining
                episodeStatsMessage.accumulatedReward = accumulatedReward
                episodeStatsMessage.accumulatedDiscountedReward = accumulatedDiscountedReward
                self.EpisodeStatsPub.publish(episodeStatsMessage)
                print("episode:", episodeNo, "number of steps:", (episodeStatsMessage.numberOfsteps),
                      "accumulatedReward:", accumulatedReward,
                      "accumulatedDiscountedReward:", accumulatedDiscountedReward)
                episodeNo += 1
                accumulatedReward = 0.0
                accumulatedDiscountedReward = 0.0
                discount = 1.0
                #print("reset")
                nextState, nextMDP, nextAs = self.domain.start()
                sleep(random.uniform(1.0, 5.0))
                self.episodeStepsRemaining = self.maxStepsPerEpisode
            else:
                nextState = deepcopy(sampleMessage.nextState)
                nextMDP = deepcopy(sampleMessage.MDP)
                nextAs = deepcopy(sampleMessage.As)

            # While calling the actionService could in theory cause the rate to drop below 1.0/self.domain.dt,
            # in practice this does not seem to be even close to happening
            nextAction = array(self.actionService(nextState, nextMDP, len(sampleMessage.action), nextAs).action)
            self.rate.sleep()    # sleep for 1.0/self.stepsPerSecond minus the time it took to execute the steps above

    def environmentDescriptionHandler(self, req):
        return EnvironmentDescriptionResponse(self.domain.stateLength, self.domain.actionLength, self.domain.MDPLength,
                                                                self.domain.gamma, self.domain.Rmax, self.domain.Qmax)

if __name__ == '__main__':
    try:
        domain = Domain()
    except rospy.ROSInterruptException:
        pass
