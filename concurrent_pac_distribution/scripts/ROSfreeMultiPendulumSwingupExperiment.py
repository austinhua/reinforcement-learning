#!/usr/bin/env python

__author__ = 'Jason Pazis'

import sys
from numpy import *
from time import time
from threading import Lock
from PendulumSwingup import PendulumSwingup
from PendulumDistance import PendulumDistance
from ConcurrentPAC import ConcurrentPAC

dKnown = float(sys.argv[1])
epsilonA = float(sys.argv[2])
epsilonB = float(sys.argv[3])
normOrder = 2
regularizer = float(sys.argv[4])*array([1.0, 1.0, 20, 50.0])

maxStepsPerEpisode = 200
episodeStepsRemaining = maxStepsPerEpisode
maxNumberOfEpisodes = int(sys.argv[5])
numberOfDomains = int(sys.argv[6])

print("dKnown", dKnown, "epsilonA", epsilonA, "epsilonB", epsilonB,
      "normOrder", normOrder, "regularizer", regularizer,
      "maxStepsPerEpisode:", maxStepsPerEpisode, "episodeStepsRemaining:", episodeStepsRemaining,
      "maxNumberOfEpisodes:", maxNumberOfEpisodes)

distance = PendulumDistance(dKnown=dKnown, normOrder=normOrder, regularizer=regularizer)

domainList = []
MDPlist = []
for i in range(numberOfDomains):
    domainList.append(PendulumSwingup())
    res1, MDP, res3 = domainList[i].start()
    MDPlist.append(MDP)
learner = ConcurrentPAC(Qmax=domainList[0].Qmax, distance=distance, gamma=domainList[0].gamma, Ka=[1, 2, 4],
                        epsilonA=epsilonA, epsilonB=epsilonB, stateLength=domainList[0].stateLength,
                        actionLength=domainList[0].actionLength, MDPlength=domainList[0].MDPLength, mutex=Lock())


for episodeNo in range(1, maxNumberOfEpisodes+1):
    accumulatedReward = []
    accumulatedDiscountedReward = []
    nextState = []
    MDP = []
    for i in range(numberOfDomains):
        res1, res2, res3 = domainList[i].start()
        domainList[i].MDP = array(MDPlist[i])
        domainList[i].m = MDPlist[i][0]
        nextState.append(res1)
        MDP.append(MDPlist[i])
        accumulatedReward.append(0.0)
        accumulatedDiscountedReward.append(0.0)
    discount = 1.0
    step = 0
    while step < maxStepsPerEpisode:
        step += 1
        for i in range(numberOfDomains):
            # Perform a single step in the domain
            state, action, reward, nextState[i], As, absorbing, MDP[i] = domainList[i].step(
                array(learner.policy.policy(nextState[i], MDP[i], domainList[i].As)))
            accumulatedReward[i] += reward
            accumulatedDiscountedReward[i] += discount*reward
            learner.addSampleToTheCandidateQueue(state, action, reward, nextState[i], domainList[i].As, absorbing, MDP[i])
        discount *= domainList[0].gamma
        learner.step()
    for i in range(numberOfDomains):
        print("episode:", episodeNo, "number of steps:", step,
                          "accumulatedReward:", accumulatedReward[i],
                          "accumulatedDiscountedReward:", accumulatedDiscountedReward[i])
    learner.printStats()
