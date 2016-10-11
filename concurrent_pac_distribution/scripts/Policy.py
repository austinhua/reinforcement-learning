__author__ = 'Jason Pazis'

from numpy import *
from Distance import Distance


class Policy(object):
    def __init__(self, Qmax, distance, regularizedU, EpsilonBK, F):
        self.Qmax = Qmax
        self.distance = distance
        self.regularizedU = regularizedU
        self.EpsilonBK = EpsilonBK
        self.F = F

    # sma must be regularized
    # and at least one approximation unit must be active
    def Qsma(self, reg_sma):
        (d, index) = self.distance.val_n_arg_min_bonus_distance(
                self.regularizedU, reg_sma, self.EpsilonBK)
        return min(self.Qmax, self.F[index]+d)

    # smaMatrix must be regularized
    def QsmaMatrix(self, reg_smaMatrix):
        return array(list(map(self.Qsma, reg_smaMatrix)))

    def Q(self, s, m, a):
        return self.Qsma(self.distance.concatenateRegularizeSma(s,m,a))

    def policy(self, s, m, As):
        return As[argmax([self.Q(s,m,a) for a in As])]

    def value(self, reg_smaList):
        return max([self.Qsma(reg_sma) for reg_sma in reg_smaList])
