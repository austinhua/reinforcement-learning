#!/usr/bin/env python

__author__ = 'Jason Pazis'

from numpy import *
import matplotlib.pyplot as plt

noEpisodes = 256
noConcurrent = 2
prefix = '/home/dev/results/ROSfreePendulumSwingupExperiment0.1.0.001.0.1.1.0.'+ str(noEpisodes)+'.'
postfix = '.out'

noIterations = 100
y = zeros(noEpisodes)
for i in range(1, noIterations+1):
    filename = prefix + str(i) + postfix
    file = open(filename)
    #yy = []
    for line in file:
        if line[2] == "e":
            line = line.split(",")
            episode = int(line[1].strip().strip(')'))-1
            #print("episode", episode)
            elem = float(line[7].strip().strip(')'))
            y[episode] += elem
            #yy.append(elem)
        # if len(yy) == 512:
        #    break
    file.close()
    # y += array(yy)

y /= float(noIterations)
y /= noConcurrent
#y = y[0:128]
x = array(range(1, len(y)+1))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Number of episodes', fontsize=16, fontweight='bold')
ax.set_ylabel('Accumulated discounted reward', fontsize=16, fontweight='bold')
plt.plot(x, y)
plt.show(block=True)
