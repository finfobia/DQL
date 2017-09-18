#/usr/local/bin/python

import tensorflow as tf
import simulator as s
import numpy as np
import random
import math


# learning rate
ETA_CRITIC = 0.0001
ETA_ACTOR = 0.000001

# decay rate of reward
GAMMA = 0.90

# CRITIC
HIDDEN_UNITS_CRITIC = 8
HIDDEN_UNITS_ACTOR = 5

# ACTOR
#NOISE = 0.5
EPSILON_ACTOR = .01
NOISE = 2

# dimensionality of input 
STATE_DIM = 1
# dimensionality of output
ACTION_DIM = 1
# HISTORY, how many past states to fold into input of network
HISTORY = 2
# 
FEATURE_DIM = STATE_DIM * (1 + HISTORY)

# REPLAY BUFFER
REPLAY_BUFFER_SIZE = 5000
SAMPLE_SIZE = 1000 

# World
TARGET = 170

# In this file we use the following syntax of local variables:
#  s : tensor flow variable
#  s_ : state, encoded in compact "gridworld" representation. i.e. coordinates, action enum
#  s__ : state, encoded as sparse ("one-hot") vector for input to neural network
#
# The functions Build*() have the side-effect of updating the global buffers in-state.

# Actor
def buildActorNetwork(s, hidden_units):
	"""Actor takes the state of the world as input and outputs the trained action.
	
	State is encoded as one-hot vector of where the agent is in the world. Action
	is a one-hot vector (softmax) of which action to take next."""

	with tf.name_scope("Actor") as scope:
		# Actor
		# Single layer network that outputs 4 directions
		W_actor = tf.Variable(tf.truncated_normal([FEATURE_DIM, hidden_units], 0, 1e-02))
		b_actor = tf.Variable(tf.truncated_normal([hidden_units],0.5,1e-01))
		hidden_actor = tf.tanh(tf.matmul(s, W_actor) + b_actor) 
	#	hidden_actor = tf.nn.relu(tf.matmul(s, W_actor) + b_actor) 
	
		Wh_actor = tf.Variable(tf.truncated_normal([hidden_units, ACTION_DIM], 0, 1e-02))
		bh_actor = tf.Variable(tf.zeros([ACTION_DIM]))
		
		y_actor = tf.matmul(hidden_actor, Wh_actor) + bh_actor
	
	tf.histogram_summary('w_actor', W_actor)
	tf.histogram_summary('wh_actor', Wh_actor)

	return y_actor

# Critic
def buildCriticNetwork(s, a, hidden_units):
	"""Critic evaluates the value of an action in a given state.

	Takes state and action as one-hot vectors as input and returns the value.
	It's currently a 2 layer network."""
	
	with tf.name_scope("Critic") as scope:
		Ws_critic = tf.Variable(tf.truncated_normal([FEATURE_DIM, hidden_units]))
		Wa_critic = tf.Variable(tf.truncated_normal([ACTION_DIM, hidden_units]))
		b_critic = tf.Variable(tf.truncated_normal([hidden_units],0.5,1e-01))
	
		hidden_critic = tf.nn.relu(tf.matmul(s, Ws_critic) + tf.matmul(a, Wa_critic) + b_critic)
	#	hidden_critic = tf.tanh(tf.matmul(s, Ws_critic) + tf.matmul(a, Wa_critic) + b_critic)
		Wh_critic = tf.Variable(tf.truncated_normal([hidden_units, 1], 0, 1e-02))
		bh_critic = tf.Variable(tf.zeros([1]))
		
		qa_critic = tf.matmul(hidden_critic, Wh_critic) + bh_critic
	tf.histogram_summary('ws_critic', Ws_critic)
	tf.histogram_summary('wa_critic', Wa_critic)
	tf.histogram_summary('wh_critic', Wh_critic)

	return qa_critic



s__ = np.zeros([REPLAY_BUFFER_SIZE, FEATURE_DIM]).astype(np.float32)
s_print__ = np.zeros([1, FEATURE_DIM]).astype(np.float32)

a__ = np.zeros([REPLAY_BUFFER_SIZE, ACTION_DIM]).astype(np.float32)
a_print__ = np.zeros([1, ACTION_DIM]).astype(np.float32)
a_eval__ = np.zeros([1, ACTION_DIM]).astype(np.float32)

r__ = np.zeros([REPLAY_BUFFER_SIZE])
t__ = np.zeros([REPLAY_BUFFER_SIZE]).astype(np.bool)

def ActionActor(network, epsilon, noise, buff):
#	a__[buff] = np.random.normal(0, 1)
#	return
#	a__[buff] = -1.0
#	return
#	a__[buff] = 0.1 * (s__[buff, HISTORY] - s__[buff, 0])
#	return
	if buff == -1:
		a_print__ = network['qa_actor'].eval(feed_dict = {network['sa']: s_print__[np.newaxis, 0, :]})
	else:
		a___ = network['qa_actor'].eval(
			feed_dict = {network['sa']: s__[np.newaxis, buff, :]})
		if np.random.rand() < epsilon:
			a__[buff] = a___ + noise* np.random.normal(0, 1, ACTION_DIM)
		else:
			a__[buff] = a___

def ActionActorVectorized(network, sample):
	a_l__ = network['qa_actor'].eval(feed_dict = {network['sa']: sample})
#	a_l__ = -1.0 * np.ones([SAMPLE_SIZE, 1])
#	a_l__ = np.random.normal(0,1,size=[SAMPLE_SIZE, 1])
#	a_l__ = (sample[np.newaxis,:,HISTORY] - sample[np.newaxis,:,0]).transpose()
	vv = network['qa_critic'].eval(feed_dict = {network['s']: sample, network['a']: a_l__})
	return vv,a_l__

# qa_print = np.zeros([FEATURE_DIM]).astype(np.float32)
# a_print = np.zeros([FEATURE_DIM])
# def PrintQAActor(network):
# 	for s_ in [(x,y) for x in range (0,WIDTH) for y in range (0,HEIGHT)]:
# 		ActionActor(network, 0, -1)
# 		q_ = network['qa_critic'].eval(feed_dict = {network['s']: s_print__[np.newaxis, 0,:], network['a']: a_print__[np.newaxis, 0, :]})
# 		a_print[s_[0] + s_[1] * WIDTH] = a_
# 		qa_print[s_[0] + s_[1] * WIDTH] = q_
# 	print qa_print.reshape(HEIGHT, WIDTH)
# 	print a_print.reshape(HEIGHT, WIDTH)
	
qa_print = np.zeros([100, 200, 2]).astype(np.float32)
def updateActorChart(state, value, buff):
	x_bucket = int(math.floor(state[0]))
	if x_bucket < 0:
		x_bucket = 0
	if x_bucket >= 200:
		x_bucket = 199
	xdot_bucket = int(math.floor(state[1]/2.0) + 50)
	if xdot_bucket < 0:
		xdot_bucket = 0
	if xdot_bucket >= 100:
		xdot_bucket = 99
	qa_print[xdot_bucket, x_bucket, buff] = value

def setupNetworks():
	# State is one-hot vector of location
	s = tf.placeholder(tf.float32, [None, FEATURE_DIM], "s")
		
	# Action is one-hot vector of action
	a = tf.placeholder(tf.float32, [None , ACTION_DIM], "a")

	qa_critic = buildCriticNetwork(s, a, HIDDEN_UNITS_CRITIC)

	sa = tf.placeholder(tf.float32, [None, FEATURE_DIM], "sa")
	qa_actor = buildActorNetwork(sa, HIDDEN_UNITS_ACTOR)

	return {'s': s, 'a': a, 'sa': sa, 'qa_critic': qa_critic, 'qa_actor': qa_actor}

# history buffer
s_hist = np.zeros([HISTORY]).astype(np.float32)
s_i = 0

# Adds new state to history and fills s[buff] with the state+history
def BuildState(s, buff, t):
	global s_i
	global s_hist
	s_i = (s_i + 1) % HISTORY
	s_hist[s_i] = s[0]
	if t:
		# if terminal, reset full state of the history
		for i in range(HISTORY):
			s_hist[i] = s[0]
	for i in range(HISTORY):
		s__[buff, i] = s_hist[(s_i - i + HISTORY) % HISTORY]
	# last input is target x value.
#	s__[buff, HISTORY] = TARGET
	s__[buff, HISTORY] = s[2]
#	print s__[1:10,:]

NBUCKETS = 201
vplot = np.zeros([NBUCKETS,3])
def updateV(ss, vs, aas, gg):
	idx = np.floor(ss[np.newaxis,:,0]).astype(np.int)
	idx[idx<0] = 0
	idx[idx>NBUCKETS] = NBUCKETS
	vplot[idx,0] = np.transpose(vs)
	vplot[idx,1] = np.transpose(aas)
	vplot[idx,2] = np.transpose(gg*10)

fig, ax = s.plt.subplots(2,1)
s.plt.show(False)


def plotV():
#	print vplot
#	s.plt.clf()
#	s.plt.plot(vplot)
#	fig.canvas.draw()
#	s.plt.show(False)
	s.plt.clf()
	s.plt.subplot(311)
	s.plt.plot(vplot)

	s.plt.subplot(312)
	s.plt.imshow(qa_print[:,:,0])
	fig.canvas.draw()

	s.plt.subplot(313)
	s.plt.imshow(qa_print[:,:,1])
	fig.canvas.draw()

	s.plt.show(False)


def Train(sess, g, network):

	a = network['a']
	s = network['s']
	sa = network['sa']
	qa_critic = network['qa_critic']

	# y will hold the inferred qa value for state s following the actors greedy policy
	y = tf.placeholder(tf.float32, [None, 1], "y")

	# critic has square loss
	loss_critic = tf.reduce_mean(tf.square(y - qa_critic))
	tf.scalar_summary('loss critic', loss_critic)

#	optimizer_critic = tf.train.GradientDescentOptimizer(ETA_CRITIC)
	optimizer_critic = tf.train.AdamOptimizer(ETA_CRITIC)
	train_critic = optimizer_critic.minimize(loss_critic)

	tv = tf.trainable_variables()

	grad = tf.gradients(network['qa_critic'], network['a'])
	gradv = tf.placeholder(tf.float32, [None, ACTION_DIM])
#	optimizer_actor = tf.train.GradientDescentOptimizer(ETA_ACTOR)
	optimizer_actor = tf.train.AdamOptimizer(ETA_ACTOR)
	gs = tf.gradients(network['qa_actor'], tv, -gradv)
#	gsclipped = [tf.clip_by_norm(ggs, 0.1) for ggs in gs]
	gs_apply = optimizer_actor.apply_gradients(zip(gs, tv))
#	reward_actor = tf.reduce_mean(tf.reduce_sum(tf.mul(gradv, network['qa_actor']),1))
#	train_actor = optimizer_actor.minimize(-reward_actor)

#	update_target_network_params = \
#    [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + \
#        tf.mul(self.target_network_params[i], 1. - self.tau))
#        for i in range(len(self.target_network_params))]
	merged = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter('/tmp/train', sess.graph)

	buffer = 0
	bufferFull = False
	terminal = True

	seq = np.array(range(REPLAY_BUFFER_SIZE))

	sess.run(tf.initialize_all_variables())

	ecounter = 0
	for step in range(200000000):
		s_ = g.getState()
		BuildState(s_, buffer, terminal)
		ActionActor(network, EPSILON_ACTOR, NOISE, buffer)
		a_ = a__[buffer]


		# Take action in the world
		s_next_, r, terminal = g.step(a_)
		v = network['qa_critic'].eval(feed_dict = {s: s__[np.newaxis, buffer, :], a: a__[np.newaxis, buffer, :]})

		updateActorChart(s_, a_, 0)
		updateActorChart(s_, v, 1)

#		print "predicted r: %f" % v

		r__[buffer] = r
		t__[buffer] = terminal
	
		next_buffer = (buffer + 1) % REPLAY_BUFFER_SIZE
		if next_buffer == 0:
			bufferFull = True

		if bufferFull:
			mask = np.ones(len(seq)).astype(np.bool)
			# don't train on the last state b/c the circular buffer will wrap around.
			mask[buffer] = False

			#randomly sample SAMPLE_SIZE samples from replay buffer
			sample_idx = random.sample(seq[mask], SAMPLE_SIZE)

			ss__ = s__[sample_idx, :]
			as__ = a__[sample_idx, :]
			rs__ = r__[sample_idx, np.newaxis]
			ts__ = t__[sample_idx, np.newaxis]


			def addoneandwrap(x):
				return (x + 1) % REPLAY_BUFFER_SIZE

			# Find next state for samples. This will be buggy when the sampled index
			# is at edge of the circular buffer, but we don't care for now.
			ss_next__ = s__[np.apply_along_axis(addoneandwrap,0,sample_idx), :]


			# Get values of greedy actions
			vv,aa = ActionActorVectorized(network, ss_next__)

			# Compute reward for each of the sample states
			ys__ = rs__ + GAMMA * vv
			# If next state is terminal, use reward only.
			ys__ = ts__ * rs__ + (1 - ts__) * ys__
	
			train_critic.run(feed_dict = {s: ss__, a: as__, y: ys__})
			

			gg = grad[0].eval(feed_dict = {s: ss__, a: as__, y: ys__})
			gs_apply.run(feed_dict = {network['sa']: ss__, gradv: gg})
#			gg = sess.run(grad, feed_dict = {s: ss__, a: as__, y: ys__})
#			sess.run(gs_apply, feed_dict = {network['sa']: ss__, gradv: gg[0]})

#			gs_fixed = [(tf.mul(gv[0], gg), gv[1]) for gv in gs if gv[0] != None]
#			train_actor.run(feed_dict = {network['sa']: ss__, gradv: gg})

			# simply for plotting the current values
			updateV(ss__, vv, aa, gg)

#		if bufferFull and (step % 5000 == 0 or terminal):
		if bufferFull and (step % 5000 < 200):
			if terminal and r > 9.9:
				print 'YIBBIE!'
			if step % 5000 == 0:
				plotV()
#				summary = sess.run([merged], feed_dict = {s: ss__, a: as__, y: ys__, sa: ss__})
#				train_writer.add_summary(summary[0], ecounter)
				ecounter += 1
			l = loss_critic.eval(feed_dict = {s: ss__, a: as__, y: ys__})
			q = qa_critic.eval(feed_dict = {s: s__[np.newaxis, buffer, :], a: a__[np.newaxis, buffer, :]})
			print "goal: %d, %f %f %s %s %s, loss: %f" % (s_[2], s_[0], s_[1], a_, r, q, l )

#		if step % 1000 == 0:
#			print "%s" % s__[np.newaxis, buffer, :]

		buffer = next_buffer
		s_ = s_next_

def RunWorld():
	sess = tf.InteractiveSession()
	network = setupNetworks()
	g = s.Simulator(80, TARGET, 10)


	Train(sess, g, network)

# Some leftovers from an early failure on actor-critic
# grad = tf.gradients(qa_critic, [a])
# reward_actor = tf.reduce_mean(tf.matmul(grad[0], tf.transpose(a_actor)))
# train_actor = optimizer.minimize(-reward_actor)

RunWorld()
