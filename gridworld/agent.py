import tensorflow as tf
import gridworld as gw
import numpy as np

#ETA = 0.005
ETA = 0.01
WIDTH = 6
HEIGHT = 2
ACTIONS = 4
GAMMA = 0.90

# CRITIC
HIDDEN_UNITS_CRITIC = 10

# EPSILON-GREEDY
EPSILON = 0.3

# ACTOR
NOISE = 0.7

# REPLAY BUFFER
REPLAY_BUFFER_SIZE = 1000
SAMPLE_SIZE = 400 

# In this file we use the following syntax of local variables:
#  s : tensor flow variable
#  s_ : state, encoded in compact "gridworld" representation. i.e. coordinates, action enum
#  s__ : state, encoded as sparse ("one-hot") vector for input to neural network
#
# The functions Build*() have the side-effect of updating the global buffers in-state.

# Actor
def buildActorNetwork(s):
	"""Actor takes the state of the world as input and outputs the trained action.
	
	State is encoded as one-hot vector of where the agent is in the world. Action
	is a one-hot vector (softmax) of which action to take next."""

	# Actor
	# Single layer network that outputs 4 directions
	W_actor = tf.Variable(tf.truncated_normal([WIDTH * HEIGHT, ACTIONS]))
	b_actor = tf.Variable(tf.zeros([ACTIONS]))
	y_actor = tf.sigmoid(tf.matmul(s, W_actor) + b_actor) 
#	Softmax is super slow. Haven't looked into why, but we can emulate an argmax.
#	y_actor = tf.nn.softmax(tf.matmul(s, W_actor) + b_actor) 

	return y_actor

# Critic
def buildCriticNetwork(s, a, hidden_units):
	"""Critic evaluates the value of an action in a given state.

	Takes state and action as one-hot vectors as input and returns the value.
	It's currently a 2 layer network."""
	
	Ws_critic = tf.Variable(tf.truncated_normal([WIDTH * HEIGHT, hidden_units]))
	Wa_critic = tf.Variable(tf.truncated_normal([ACTIONS, hidden_units]))
	b_critic = tf.Variable(tf.zeros([hidden_units]))

	hidden_critic = tf.tanh(tf.matmul(s, Ws_critic) + tf.matmul(a, Wa_critic) + b_critic)
	Wh_critic = tf.Variable(tf.truncated_normal([hidden_units, 1], 0, 1e-02))
	bh_critic = tf.Variable(tf.zeros([1]))
	
	qa_critic = tf.tanh(tf.matmul(hidden_critic, Wh_critic) + bh_critic)

	return qa_critic



s__ = np.zeros([REPLAY_BUFFER_SIZE, WIDTH * HEIGHT]).astype(np.float32)
s_print__ = np.zeros([1, WIDTH * HEIGHT]).astype(np.float32)
def BuildState(s, buff):
	if buff == -1:
		s_print__[0, :] = 0.0
		s_print__[0, s[0] + s[1] * WIDTH] = 1.0
		return		
	s__[buff, :] = 0.0
	s__[buff, s[0] + s[1] * WIDTH] = 1.0

# Using two action buffers for a_i and a_{i+1}
a__ = np.zeros([REPLAY_BUFFER_SIZE, ACTIONS]).astype(np.float32)
a_print__ = np.zeros([1, ACTIONS]).astype(np.float32)
a_eval__ = np.zeros([1, ACTIONS]).astype(np.float32)
def BuildAction(action, buff):
	# special value -1 for buff is the print buffer
	if buff == -1:
		a_print__[0, :] = 0.0
		a_print__[0, action] = 1.0
		return
	# special value -2 for buff is the tmp buffer for computing greedy action.
	if buff == -2:
		a_eval__[0, :] = 0.0
		a_eval__[0, action] = 1.0
		return
	a__[buff, :] = 0.0
	a__[buff, action] = 1.0

r__ = np.zeros([REPLAY_BUFFER_SIZE])
t__ = np.zeros([REPLAY_BUFFER_SIZE]).astype(np.bool)

def ActionEpsilonGreedy(network, epsilon, buff):
	"""Finds epsilon-greedy action for state in buff and returns action.

	Alters state of action buffer in buff"""
	# Greedy action?
	if np.random.rand() < epsilon:
		am = np.random.randint(ACTIONS)
	else:
		am = 0
		qm = -1000
		for i in range(ACTIONS):
			BuildAction(i, buff)
			if buff == -1:
				v = network['qa_critic'].eval(feed_dict = {network['s']: s_print__[np.newaxis, 0, :], network['a']: a_print__[np.newaxis, 0, :]})
			else:
				v = network['qa_critic'].eval(feed_dict = {network['s']: s__[np.newaxis, buff, :], network['a']: a__[np.newaxis, buff, :]})
			if v > qm:
				am = i
				qm = v
	BuildAction(am, buff)
	return am

def ActionGreedyVectorized(network, sample):
	v = np.zeros([SAMPLE_SIZE, ACTIONS]).astype(np.float32)
	for i in range(ACTIONS):
		BuildAction(i, -2)
		k = network['qa_critic'].eval(feed_dict = {network['s']: sample, network['a']: a_eval__[np.newaxis, 0, :]})
		v[:,i] = k.flatten()
	vv = np.amax(v, axis=1).reshape([SAMPLE_SIZE, 1])
	return vv

def ActionActor(network, noise, buff):
	if buff == -1:
		a = network['qa_actor'].eval(feed_dict = {network['sa']: s_print__[np.newaxis, 0, :]})
	else:
		a = network['qa_actor'].eval(feed_dict = {network['sa']: s__[np.newaxis, buff, :]})
	a_ = np.argmax(a + noise * np.random.normal(0,1,ACTIONS))

	if buff == -1:
		print a
	BuildAction(a_, buff)
	return a_

def ActionActorVectorized(network, sample):
	a = network['qa_actor'].eval(feed_dict = {network['sa']: sample})
	a_ = np.argmax(a, axis=1)

	a__ = np.zeros([SAMPLE_SIZE, ACTIONS]).astype(np.float32)
	t = range(SAMPLE_SIZE)
	a__[t, a_] = 1.0
	vv = network['qa_critic'].eval(feed_dict = {network['s']: sample, network['a']: a__})
	return vv

qa_print = np.zeros([WIDTH * HEIGHT]).astype(np.float32)
a_print = np.zeros([WIDTH * HEIGHT])
def PrintQA(network):
	"""Print QA values for each possible state.

	Alters buffer 1 of s__ and a__."""
	for s_ in [(x,y) for x in range (0,WIDTH) for y in range (0,HEIGHT)]:
		BuildState(s_, -1)
		# Find argmax action on state s_
		a_ = ActionEpsilonGreedy(network, 0, -1)
		# Evaluate network to give expected reward / value of s_ following trained policy
		q_ = network['qa_critic'].eval(feed_dict = {network['s']: s_print__[np.newaxis, 0,:], network['a']: a_print__[np.newaxis, 0,:]})
		a_print[s_[0] + s_[1] * WIDTH] = a_
		qa_print[s_[0] + s_[1] * WIDTH] = q_
	print qa_print.reshape(HEIGHT, WIDTH)
	print a_print.reshape(HEIGHT, WIDTH)

def PrintQAActor(network):
	for s_ in [(x,y) for x in range (0,WIDTH) for y in range (0,HEIGHT)]:
		BuildState(s_, -1)
		a_ = ActionActor(network, 0, -1)
		q_ = network['qa_critic'].eval(feed_dict = {network['s']: s_print__[np.newaxis, 0,:], network['a']: a_print__[np.newaxis, 0, :]})
		a_print[s_[0] + s_[1] * WIDTH] = a_
		qa_print[s_[0] + s_[1] * WIDTH] = q_
	print qa_print.reshape(HEIGHT, WIDTH)
	print a_print.reshape(HEIGHT, WIDTH)
	

def setupNetworks():
	# State is one-hot vector of location
	s = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT], "s")
		
	# Action is one-hot vector of action
	a = tf.placeholder(tf.float32, [None , ACTIONS], "a")

	qa_critic = buildCriticNetwork(s, a, HIDDEN_UNITS_CRITIC)

	sa = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT], "sa")
	qa_actor = buildActorNetwork(sa)

	return {'s': s, 'a': a, 'sa': sa, 'qa_critic': qa_critic, 'qa_actor': qa_actor}

def Train(g, network):
	a = network['a']
	s = network['s']
	qa_critic = network['qa_critic']

	# y will hold the inferred qa value for state s following the actors greedy policy
	y = tf.placeholder(tf.float32, [None, 1], "y")

	# critic has square loss
	loss_critic = tf.reduce_mean(tf.square(y - qa_critic))
	optimizer = tf.train.GradientDescentOptimizer(ETA)
	train_critic = optimizer.minimize(loss_critic)

	grad = tf.gradients(network['qa_critic'], network['a'])
	gradv = tf.placeholder(tf.float32, [None, ACTIONS])
	reward_actor = tf.reduce_mean(tf.reduce_sum(tf.mul(gradv, network['qa_actor']),1))
	train_actor = optimizer.minimize(-reward_actor)

	buffer = 0
	bufferFull = False
	for step in range(200000000):
		s_ = g.getState()
		# Use buffer 0 for state s_i
		BuildState(s_, buffer)
		# Find epsilon-greedy action for state s_
#		a_ = ActionEpsilonGreedy(network, EPSILON, buffer)
		a_ = ActionActor(network, NOISE, buffer)

		# Take action in the world
		s_next_, r, terminal = g.step(a_)

		r__[buffer] = r
		t__[buffer] = terminal
	
		next_buffer = (buffer + 1) % REPLAY_BUFFER_SIZE
		if next_buffer == 0:
			bufferFull = True

		if bufferFull:
			#randomly sample SAMPLE_SIZE samples from replay buffer
			sample_idx = np.random.choice(REPLAY_BUFFER_SIZE, SAMPLE_SIZE)

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
#			vv = ActionGreedyVectorized(network, ss_next__)
			vv = ActionActorVectorized(network, ss_next__)

			# Compute reward for each of the sample states
			ys__ = rs__ + GAMMA * vv
			# If next state is terminal, use reward only.
			ys__ = ts__ * rs__ + (1 - ts__) * ys__
	
			gg = grad[0].eval(feed_dict = {s: ss__, a: as__, y: ys__})
#			print "grad"
#			print gg
			train_critic.run(feed_dict = {s: ss__, a: as__, y: ys__})
			train_actor.run(feed_dict = {network['sa']: ss__, gradv: gg})
	
		if step % 1000 == 0 or terminal:
			q = qa_critic.eval(feed_dict = {s: s__[np.newaxis, buffer, :], a: a__[np.newaxis, buffer, :]})
			print "%s %s %s %s" % (s_, a_, r, q)
#			PrintQA(network)
			PrintQAActor(network)

		buffer = next_buffer
		s_ = s_next_

def RunWorld():
	sess = tf.InteractiveSession()
	network = setupNetworks()
#	g = gw.GridWorld(WIDTH, HEIGHT, (WIDTH-1, HEIGHT-1), (int(WIDTH/2), 100))
	g = gw.GridWorld(WIDTH, HEIGHT, (WIDTH-1, 0), (int(WIDTH/2), 0))
#	g = gw.GridWorld(WIDTH, HEIGHT, (WIDTH-1, int(HEIGHT/2)), (int(WIDTH/2), int(HEIGHT/2)))
#	g = gw.GridWorld(WIDTH, HEIGHT, (WIDTH-1, HEIGHT-1), (WIDTH-1, 0))
	sess.run(tf.initialize_all_variables())

	Train(g, network)

# Some leftovers from an early failure on actor-critic
# grad = tf.gradients(qa_critic, [a])
# reward_actor = tf.reduce_mean(tf.matmul(grad[0], tf.transpose(a_actor)))
# train_actor = optimizer.minimize(-reward_actor)

#RunWorld()
