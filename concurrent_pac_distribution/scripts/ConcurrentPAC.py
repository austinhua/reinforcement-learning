__author__ = 'Jason Pazis'

from numpy import *
from Policy import Policy
from time import time
from Queue import Queue

class ConcurrentPAC(object):
    def __init__(self, Qmax, distance, gamma, Ka, epsilonA, epsilonB, stateLength, actionLength, MDPlength, mutex):
        # TODO specify types
        self.Qmax = Qmax
        self.distance = distance
        self.gamma = gamma
        self.Ka = Ka
        self.epsilonA = epsilonA
        self.epsilonB = epsilonB
        self.sampleCandidateQueue = Queue()
        self.mutex = mutex
        #self.mutex = Lock()

        # sample description
        self.regularizedSMA = []        # regularized state-action-MDP
        self.R = []                     # reward
        self.absorbing = []             # is this an absorbing sample?
        self.regularizedNextSMA = []    # next state-action-MDPs
        self.nextSMApointers = []       # pointers to approximation units for next state-action-MDPs
        self.nextSMAdistances = []      # distances to approximation units for next state-action-MDPs

        # approximation unit description
        self.regularizedU = []          # regularized state-action-MDP
        self.F = []                     # F value
        self.ka = []                    # number of active samples
        self.EpsilonBK = []             # bonus over square root of ka
        self.samplePointers = []        # list of all samples within dknown
        self.updateNN = []              # stack of approximation units whose NN need updating
        self.QisCurrent = True
        self.startTime = time()
        self.iterationsUntilConvergence = 0

        # Initialize approximation units with a dummy approximation unit
        # at infinite distance from non-dummy samples
        # (no samples will ever be added to this approximation unit)
        # This dummy approximation unit ensures that a nearest neighbor for
        # nextState-action-MDP triples and for policy evaluation purposes always exists
        dummy_sma = empty(stateLength + actionLength + MDPlength)
        dummy_sma.fill(inf)
        self.addApproximationUnit(dummy_sma)
        # set oldU to U
        self.syncApproximationUnits()

    def printApproximationUnits(self):
        print("printApproximationUnits")
        print("regularizedU", self.regularizedU)
        print("F", self.F)
        print("ka", self.ka)
        print("EpsilonBK", self.EpsilonBK)
        print("samplePointers", self.samplePointers)

    def printSamples(self):
        print("printSamples")
        print(self.regularizedSMA)
        print(self.R)
        print(self.regularizedNextSMA)
        print(self.nextSMApointers)
        print(self.nextSMAdistances)

    def addSampleToTheCandidateQueue(self, state, action, reward, nextState, As, absorbing, MDP):
        # sampleCandidateQueue is (supposed to be) thread safe
        self.mutex.acquire()
        self.sampleCandidateQueue.put([state, action, reward, nextState, As, absorbing, MDP])
        self.mutex.release()

    def processQueue(self):
        self.mutex.acquire()
        isEmpty = self.sampleCandidateQueue.empty()
        self.mutex.release()
        while not isEmpty:
            self.mutex.acquire()
            sample = self.sampleCandidateQueue.get()
            isEmpty = self.sampleCandidateQueue.empty()
            self.mutex.release()
            self.processSample(sample[0],sample[1], sample[2], sample[3], sample[4], sample[5], sample[6])

    # Process a sample from the queue. This may or may not lead
    # to the addition of a new approximation unit, the addition
    # of a sample to the sample set, and the addition of pointers
    # to the new sample from existing approximation units
    # Single thread function
    def processSample(self, state, action, r, nextState, As, absorbing, MDP):
        reg_sma = self.distance.concatenateRegularizeSma(state, MDP, action)

        # compute the distance of the sample's originating state-action-MDP
        # to all approximation units
        distanceVector = self.distance.distance(self.regularizedU, reg_sma)

        # if no approximation unit exists within dKnown distance,
        # add a new approximation unit
        if min(distanceVector) > 0.0: # dKnown has already been subtracted from distanceVector
            self.addApproximationUnit(reg_sma)
            distanceVector = append(distanceVector, 0.0)

        # add the sample to non-full approximation units within dKnown distance
        sampleAdded = False
        for index, d in enumerate(distanceVector):
            if d == 0.0 and self.ka[index] < self.Ka[-1]:
                #print("adding sample", len(self.R), "to approximation unit", index)
                sampleAdded = True
                self.samplePointers[index].append(len(self.R))
                noSamplePointers = len(self.samplePointers[index])
                kaNew = 0
                for ka in self.Ka:
                    if noSamplePointers >= ka:
                        kaNew = ka
                if kaNew != self.ka[index]:
                    # If the number of active samples has changed for at least one approximation unit
                    # Q needs to be recalculated
                    if self.QisCurrent:
                        # start counting time and iterations from the moment
                        # Q stops being current
                        self.QisCurrent = False
                        self.startTime = time()
                        self.iterationsUntilConvergence = 0
                    self.ka[index] = kaNew
                    self.EpsilonBK[index] = self.epsilonB/sqrt(kaNew)
                    self.updateNN.append(index)
                    #self.updateNN[index] = True

        # if the sample was added to at least one approximation unit,
        # add it to the sample set
        if sampleAdded:
            self.addSample(reg_sma, r, nextState, As, absorbing, MDP)

    # Add an approximation unit to the approximation set
    def addApproximationUnit(self, reg_sma):
        self.regularizedU.append(reg_sma)
        self.F.append(self.Qmax)
        self.ka.append(0)
        self.EpsilonBK.append(self.Qmax)
        self.samplePointers.append([])
        #self.updateNN.append(False)

    # Add a sample to the sample set
    def addSample(self, reg_sma, r, nextState, As, absorbing, MDP):
        self.regularizedSMA.append(reg_sma)
        self.R.append(r)
        self.absorbing.append(absorbing)
        next_sma_list = []
        next_sma_pointers = []
        next_sma_distances = []
        # if this is not an absorbing sample, process its nextState-action-MDP triples
        if not absorbing:
            # for all actions available in the nextState
            for a in As:
                # save nextState-action-MDP description
                next_sma = self.distance.concatenateRegularizeSma(nextState, MDP, a)
                next_sma_list.append(next_sma)
                # find and save the nearest neighbor and its distance
                dist, index = self.distance.val_n_arg_min_bonus_distance(self.policy.regularizedU, next_sma, self.policy.EpsilonBK)
                next_sma_pointers.append(index)
                next_sma_distances.append(dist)
        self.regularizedNextSMA.append(next_sma_list)
        self.nextSMApointers.append(next_sma_pointers)
        self.nextSMAdistances.append(next_sma_distances)

    # Update the policy used internally by the learner for evaluating next-state-MDPs
    # to use the updated approximation units (set oldU to U)
    def syncApproximationUnits(self):
        #print("entering syncApproximationUnits()")
        # policy takes the role of oldU
        self.mutex.acquire()
        self.policy = Policy(self.Qmax, self.distance, array(self.regularizedU).copy(), array(self.EpsilonBK).copy(),
                             array(self.F).copy())
        #print("past policy update in syncApproximationUnits()")
        self.mutex.release()

        # Check if any NN need to be updated
        if len(self.updateNN) > 0:
            # This enumeration may be used repeatedly
            enum1 = enumerate(self.regularizedNextSMA)
            # Package all next-state-action-MDP triples into one array,
            # so that distance (and the norm it uses) can be called once
            # per Uindex
            flatRegularizedNextSMA = []
            for reg_nextSMAlist in self.regularizedNextSMA:
                    for reg_nextSMA in reg_nextSMAlist:
                        flatRegularizedNextSMA.append(reg_nextSMA)
            for Uindex in self.updateNN:
                #print("updating index", Uindex)
                d = self.distance.bonus_distance(self.regularizedU[Uindex], array(flatRegularizedNextSMA), self.EpsilonBK[Uindex])
                dIndex = 0
                for sampleIndex, reg_nextSMAlist in enum1:
                    # This is necessary because the number of actions per state can vary
                    for action in range(len(reg_nextSMAlist)):
                        if d[dIndex] < self.nextSMAdistances[sampleIndex][action]:
                            self.nextSMAdistances[sampleIndex][action] = d[dIndex]
                            self.nextSMApointers[sampleIndex][action] = Uindex
                        dIndex += 1
            self.updateNN = []
        #print("exiting syncApproximationUnits()")

    # Perform a single step of value iteration
    def VIstep(self):
        # compute sample values based on approximation unit values
        def calculateX(i):
            if self.absorbing[i]:
                return self.R[i]
            else:
                Vnext = -inf
                for a in range(len(self.nextSMApointers[i])):
                    Vnext = max(Vnext, self.policy.F[self.nextSMApointers[i][a]] + self.nextSMAdistances[i][a])
                Vnext = min(self.Qmax, Vnext)
                return self.R[i] + self.gamma * Vnext
        X = list(map(calculateX, range(len(self.R))))

        # compute approximation unit values based on sample values
        F = zeros_like(self.F)
        def setF(i):
            for j in range(self.ka[i]):
                F[i] += X[self.samplePointers[i][j]]
        list(map(setF, range(len(F))))
        self.F = F / self.ka + self.EpsilonBK
        self.F = self.F.tolist()

        # Compute an upper bound on the magnitude of the Bellman error
        # Note that since this is an upper bound, it may cause us to do
        # a few more iterations than necessary, but its not clear that
        # the computational effort required for the exact Bellman error
        # is worth the price
        if len(self.F) == len(self.policy.F):
            oldValues = zeros_like(self.F)
            newValues = zeros_like(self.F)
            def calculateBellman(Findex):
                # Calculate Bellman error only if this is an active
                # approximation unit
                if self.ka[Findex] > 0:
                    oldValues[Findex] = min(self.Qmax, self.policy.F[Findex] + self.policy.EpsilonBK[Findex])
                    newValues[Findex] = min(self.Qmax, self.F[Findex] + self.EpsilonBK[Findex])
            list(map(calculateBellman, range(len(self.F))))
            BellmanErrorMagnitude = max(abs(oldValues-newValues))
        else:
            BellmanErrorMagnitude = self.Qmax

        # set oldU to U
        self.syncApproximationUnits()

        #print("BellmanErrorMagnitude", BellmanErrorMagnitude)
        return BellmanErrorMagnitude

    # returns False when Q does not need to be updated
    def step(self):
        self.processQueue()
        if self.QisCurrent:
            return False
        else:
            self.iterationsUntilConvergence += 1
            BellmanErrorMagnitude = self.VIstep()
            if BellmanErrorMagnitude <= self.epsilonA:
                self.QisCurrent = True
                # elapsedTime = time()-self.startTime
                # print("number of approximation units:", len(self.F), "number of samples:", len(self.R),
                #       "time to convergence:", elapsedTime, "iterationsUntilConvergence:",
                #       self.iterationsUntilConvergence)
            return True

    def runToConvergence(self):
        while self.step():
            pass

    def printStats(self):
        print("number of approximation units:", len(self.F), "number of samples:", len(self.R))
