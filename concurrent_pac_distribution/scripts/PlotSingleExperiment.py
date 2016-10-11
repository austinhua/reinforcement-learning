#!/usr/bin/env python

__author__ = 'Jason Pazis'

import sys
import numpy as np
import matplotlib.pyplot as plt
import os

inputfiles = [open('../results/ROSfreePendulumSwingupExperiment0.0001.0.001.0.1.0.01.128.out'),
              open('../results/ROSfreePendulumSwingupExperiment0.0001.0.001.0.1.0.1.128.out'),
              open('../results/ROSfreePendulumSwingupExperiment0.0001.0.001.0.1.1.0.128.out'),
              open('../results/ROSfreePendulumSwingupExperiment0.0001.0.001.0.1.10.0.128.out'),
              open('../results/ROSfreePendulumSwingupExperiment0.0001.0.001.0.1.100.0.128.out')]

for file in inputfiles:
    y = []
    for line in file:
        if line[2] == "e":
            elem = line.split(",")[7].strip().strip(')')
            y.append(elem)
    x = np.array(range(1, len(y)+1))
    y = np.array(y)
    fig = plt.figure()
    plt.plot(x, y)
    plt.show(block=False)
    file.close()

raw_input()