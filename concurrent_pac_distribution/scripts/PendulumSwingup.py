#!/usr/bin/env python

__author__ = 'Jason Pazis'

from numpy import *
from copy import deepcopy
from Pendulum import Pendulum

class PendulumSwingup(Pendulum):

    def reward(self, nextState, action):
        theta = nextState[0]
        if abs(theta) > (pi/2):
            return 0.0, False
        else:
            r = 1 - ((theta/pi)**2)
            return r, False

    def step(self, action):
        #print("self.m is:", self.m)
        u = action[0] + random.uniform(-1.0, 1.0)*self.noise
        previousState = self.state.copy()
        self.rk4(self.state, u, self.dt/10.0, 0, self.dt)
        # This is a while and not an if to account for
        # extreme velocities/really long timesteps
        while self.state[0] > pi:
            self.state[0] -= 2*pi
        while self.state[0] < -pi:
            self.state[0] += 2*pi
        r, absorbing = self.reward(self.state, action)
        sample = [previousState.tolist(), action.copy().tolist(), r, self.state.copy().tolist(),
                  deepcopy(self.flatAs), absorbing, self.MDP.copy().tolist()]
        #print(sample)
        return sample

    def start(self):
        theta = random.uniform(-0.2, 0.2)+pi    # angle
        theta_dot = 0                           # angular velocity
        self.state = array([theta, theta_dot])
        while self.state[0] > pi:
            self.state[0] -= 2*pi
        while self.state[0] < -pi:
            self.state[0] += 2*pi
        #self.m = random.uniform(1.0, 10.0)       # Randomize the mass of the pendulum
        #print("self.m is:", self.m)
        self.MDP = array([self.m])
        return self.state.copy().tolist(), self.MDP.copy().tolist(), deepcopy(self.flatAs)
