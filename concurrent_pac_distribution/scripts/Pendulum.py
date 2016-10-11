#!/usr/bin/env python

__author__ = 'Jason Pazis'

from numpy import *
from copy import deepcopy

class Pendulum(object):
    def __init__(self):
        self.dt = 0.1
        self.stateLength = 2
        self.actionLength = 1
        self.MDPLength = 1
        self.gamma = 0.98
        self.Rmax = 1.0
        self.Qmax = self.Rmax/(1.0-self.gamma)
        self.As = [[-50], [0], [50]]
        self.flatAs = [item for sublist in self.As for item in sublist]
        self.noise = 25.0

        self.m = 2.0            # Mass of the pendulum
        self.M = 8.0            # Mass of the cart
        self.l = 1.0/2.0        # Length of the pendulum
        self.g = 9.8            # Gravity constant
        self.a = 1.0/(self.m+self.M)

        self.MDP = array([self.m])
        self.state = array([0, 0])
        self.start()

    def f(self, state, u):
        theta = state[0]        # angle
        theta_dot = state[1]    # angular velocity
        costheta = cos(theta)
        # angular acceleration?
        theta_dot_dot = (self.g*sin(theta) - self.a*self.m*self.l*theta_dot*theta_dot*sin(2.0*theta)/2.0 - self.a*costheta*u) / (4.0*self.l/3.0 - self.a*self.m*self.l*costheta*costheta)
        return array([theta_dot, theta_dot_dot])

    def rk4(self, state, action, h, t_init, t_final):
        t = t_init
        while t < t_final:
            k1 = self.f(state,                action)
            k2 = self.f(state + (k1 * (h/2.0)), action)
            k3 = self.f(state + (k2 * (h/2.0)), action)
            k4 = self.f(state + (k3*h),       action)

            state += (k1 + k2*2.0 + k3*2.0 + k4)*h / 6.0
            t += h

    # def reward(self, nextState, action):
    #     theta = nextState[0]
    #     if abs(theta) > (pi/2.0):
    #         return 0.0, True
    #     else:
    #         return 1.0, False
    def reward(self, nextState, action):
        theta = nextState[0]
        if abs(theta) > (pi/2):
            return 0.0, True
        else:
            r = 1 - ((theta/pi)**2)
            return r, False

    def step(self, action):
        u = action[0] + random.uniform(-1.0, 1.0)*self.noise
        previousState = self.state.copy()
        self.rk4(self.state, u, self.dt/10.0, 0, self.dt)
        r, absorbing = self.reward(self.state, action)
        sample = [previousState.tolist(), action.copy().tolist(), r, self.state.copy().tolist(),
                  deepcopy(self.flatAs), absorbing, self.MDP.copy().tolist()]
        return sample

    def start(self):
        theta = random.uniform(-0.2, 0.2)       # angle
        theta_dot = random.uniform(-0.2, 0.2)   # angular velocity
        self.state = array([theta, theta_dot])
        #self.m = random.uniform(2.0, 4.0)       # Randomize the mass of the pendulum
        self.MDP = array([self.m])
        return self.state.copy().tolist(), self.MDP.copy().tolist(), deepcopy(self.flatAs)
