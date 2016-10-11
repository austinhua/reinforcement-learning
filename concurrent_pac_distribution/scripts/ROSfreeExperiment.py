#!/usr/bin/env python

__author__ = 'Jason Pazis'

import sys
from numpy import *
from time import time
from threading import Lock
from Pendulum import Pendulum
from HIVtreatment import HIVtreatment
from Distance import Distance
from ConcurrentPAC import ConcurrentPAC

dKnown = float(sys.argv[1])
epsilonA = float(sys.argv[2])
epsilonB = float(sys.argv[3])
normOrder = 2
regularizer = float(sys.argv[4])*array([1.0, 1.0, 20, 50.0])

maxStepsPerEpisode = 200
episodeStepsRemaining = maxStepsPerEpisode
maxNumberOfEpisodes = int(sys.argv[5])

print("dKnown", dKnown, "epsilonA", epsilonA, "epsilonB", epsilonB,
      "normOrder", normOrder, "regularizer", regularizer,
      "maxStepsPerEpisode:", maxStepsPerEpisode, "episodeStepsRemaining:", episodeStepsRemaining,
      "maxNumberOfEpisodes:", maxNumberOfEpisodes)

distance = Distance(dKnown=dKnown, normOrder=normOrder, regularizer=array(regularizer))

domain = HIVtreatment()
learner = ConcurrentPAC(Qmax=domain.Qmax, distance=distance, gamma=domain.gamma, Ka=[1], epsilonA=epsilonA,
                        epsilonB=epsilonB, stateLength=domain.stateLength, actionLength=domain.actionLength,
                        MDPlength=domain.MDPLength, mutex=Lock())

for episodeNo in range(1, maxNumberOfEpisodes+1):
    nextState, nextMDP, nextAs = domain.start()
    accumulatedReward = 0.0
    accumulatedDiscountedReward = 0.0
    discount = 1.0
    step = 0
    while step < maxStepsPerEpisode:
        step += 1
        # Perform a single step in the domain
        state, action, reward, nextState, As, absorbing, MDP = domain.step(array(learner.policy.policy(nextState, nextMDP,
                                                                                                 domain.As)))
        if episodeNo == maxNumberOfEpisodes:
            print("state", state, "action", action, "reward", reward, "nextState", nextState, "As", As,
                  "absorbing", absorbing, "MDP", MDP)
        accumulatedReward += reward
        accumulatedDiscountedReward += discount*reward
        discount *= domain.gamma
        learner.addSampleToTheCandidateQueue(state, action, reward, nextState, domain.As, absorbing, MDP)
        #learner.runToConvergence()
        learner.step()
        if absorbing:
            break
    print("episode:", episodeNo, "number of steps:", step,
                      "accumulatedReward:", accumulatedReward,
                      "accumulatedDiscountedReward:", accumulatedDiscountedReward)
