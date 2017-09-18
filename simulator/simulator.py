import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from matplotlib import animation

import math
import random

TOP = 200
FORCE_CAP = 100

class Simulator:
	def __init__(self, initheight, target, sigma):
		self.initheight = initheight
		self.tau = 0.05
		self.m = 10
		self.target = target
		self.sigma = sigma
		self.targetcounter = 0
		self.reset()
#		self.initGraphics()
		self.line = 0

	def initGraphics(self):
		self.fig, self.ax = plt.subplots(1,1)
		self.ax.set_xlim(0, 20)
		self.ax.set_ylim(-10, TOP)
		self.ax.hold()

		plt.plot([0, 20], [self.target, self.target], 'r')
		plt.plot([0, 20], [0, 0], 'k')

		plt.show(False)


	def plot(self):
		print "bb"

		if self.line != 0:
			self.line.pop(0).remove()
		self.ax.set_ylim(-10, TOP)

		self.line = plt.plot([5, 15], [self.state[0], self.state[0]], 'b')
		self.fig.canvas.draw()

	def reset(self):
		# (x, xdot)
		self.target = random.randint(1, TOP-1)
		self.state = (random.randint(1,TOP-1), 0, self.target)

	def getState(self):
		return self.state

	def step(self, force):
		# cap force at FORCE_CAP
		if math.fabs(force) > FORCE_CAP:
			force = math.copysign(FORCE_CAP, force)
		xdot = self.state[1] + self.tau * self.m * force
#		xdot = 10 * force
		x = self.state[0] + self.tau * xdot
		self.state = (x, xdot, self.target)

		t = False
		r = 0
		if x < 0 or x > TOP :
			r = -10.0 
			t = True
		else:
			r = math.exp(-(x - self.target)**2 / (2 * self.sigma**2)) * 2.0

		if math.fabs(x - self.target) < self.sigma:
			self.targetcounter += 1
		else:
			self.targetcounter = 0

		if self.targetcounter > 100:
			self.targetcounter = 0
			r = 10.0
			t = True

#		print "r: %f" % r

		if t:
			self.reset()

		return self.state, r, t
