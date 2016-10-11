__author__ = 'Jason Pazis'

from numpy import *
from numpy import linalg as LA


class Distance(object):
    def __init__(self, dKnown, normOrder, regularizer):
        self.dKnown = dKnown
        self.normOrder = normOrder
        self.regularizer = regularizer

    def regularize(self, sma):
        return sma/self.regularizer

    def concatenateRegularizeSma(self, state, MDP, action):
        return concatenate((state, MDP, action))/self.regularizer

    def distance(self, SMA, sma):
        #print("SMA", SMA, "sma", sma)
        return maximum(0.0, LA.norm(transpose(SMA - sma), ord=self.normOrder, axis=0) - self.dKnown)

    def bonus_distance(self, SMA, sma, EpsilonBK):
        return self.distance(SMA, sma)+EpsilonBK

    # returns (0, 0) in case of error
    def val_n_arg_min_bonus_distance(self, SMA, sma, EpsilonBK):
        bd = self.bonus_distance(SMA, sma, EpsilonBK)
        try:
            index = nanargmin(bd)
            return (bd[index], index)
        except ValueError:
            print("got a ValueError, SMA", SMA, "sma", sma)
            return (0, 0)
