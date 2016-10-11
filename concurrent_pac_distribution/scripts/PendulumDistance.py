__author__ = 'Jason Pazis'

from numpy import *
from numpy import linalg as LA
from Distance import Distance

class PendulumDistance(Distance):

    def distance(self, SMA, sma):
        SMA_sma = SMA - sma

        def keepInRange(i):
            if SMA_sma[i,0] > pi:
                SMA_sma[i,0] -= 2*pi
            elif SMA_sma[i,0] < -pi:
                SMA_sma[i,0] += 2*pi
        list(map(keepInRange, range(len(SMA_sma))))
        return maximum(0.0, LA.norm(transpose(SMA_sma), ord=self.normOrder, axis=0) - self.dKnown)
