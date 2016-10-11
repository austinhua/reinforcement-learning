#!/usr/bin/env python

__author__ = 'Jason Pazis'

import sys
import numpy as np
import matplotlib.pyplot as plt

inputfile = open('../results/ROSfreePendulumSwingupExperiment0.0001.0.001.0.1.1.0.128.out')
outputfile = open('discAccumReward.txt', 'w')


res = []
for line in inputfile:
    if line[2] == "e":
        elem = line.split(",")[7].strip().strip(')')
        res.append(elem)
        outputfile.write(elem)
        outputfile.write('\n')

XX = np.array(range(1, len(res)+1))
YY = np.array(res)

plt.plot(XX, YY)

plt.show()

inputfile.close()
outputfile.close()
