import tensorflow as tf
import simulator as s
import numpy as np
import random


# learning rate
ETA_CRITIC = 0.00001
ETA_ACTOR = 0.00001

# decay rate of reward
GAMMA = 0.90

# CRITIC
HIDDEN_UNITS_CRITIC = 8
HIDDEN_UNITS_ACTOR = 5

# ACTOR
#NOISE = 0.5
EPSILON_ACTOR = .01
NOISE = 10

# dimensionality of input 
STATE_DIM = 1
# dimensionality of output
ACTION_DIM = 1
# HISTORY, how many past states to fold into input of network
HISTORY = 6
# 
FEATURE_DIM = STATE_DIM * (1 + HISTORY)

# REPLAY BUFFER
REPLAY_BUFFER_SIZE = 4000
SAMPLE_SIZE = 100 

# World
TARGET = 170

class Actor:
	def __init__(self, sess, feature_dim, hidden_units, action_dim, eta):
		self.vars = None
		self.sess = sess
		self.hidden_units = hidden_units
		self.feature_dim = feature_dim
		self.action_dim = action_dim
		self.s, self.a = self.buildNetwork()

		self.trainable_variables = tf.trainable_variables()

		self.gradients = tf.placeholder(tf.float32, [None, action_dim])
		self.grads = tf.gradients(self.a, self.trainable_variables, -self.gradients)
		self.optimizer = tf.train.AdamOptimizer(eta)
		self.grad_apply = self.optimizer.apply_gradients(zip(self.grads, self.trainable_variables))
	
	def buildNetwork(self):
		with tf.name_scope("Actor") as scope:
			s = tf.placeholder(tf.float32, [None, self.feature_dim], "s")

			W_actor = tf.Variable(tf.truncated_normal([self.feature_dim, self.hidden_units], 0, 1e-02))
			b_actor = tf.Variable(tf.truncated_normal([self.hidden_units],0.5,1e-01))
			self.hidden_actor = tf.tanh(tf.matmul(s, W_actor) + b_actor) 
		#	self.hidden_actor = tf.nn.relu(tf.matmul(s, W_actor) + b_actor) 
		
			Wh_actor = tf.Variable(tf.truncated_normal([self.hidden_units, self.action_dim], 0, 1e-02))
			bh_actor = tf.Variable(tf.zeros([self.action_dim]))
			
			a = tf.matmul(hidden_actor, Wh_actor) + bh_actor
		return s, a


	def actionActor(self, s):
		return self.sess.run(self.y_actor,
				feed_dict = {self.s: s})
#	a_l__ = -1.0 * np.ones([SAMPLE_SIZE, 1])
#	a_l__ = np.random.normal(0,1,size=[SAMPLE_SIZE, 1])
#	a_l__ = (sample[np.newaxis,:,HISTORY] - sample[np.newaxis,:,0]).transpose()

	def train(self, s, grad):
		self.sess.run(self.grad_apply,
				feed_dict = {self.s: s, self.gradients: gradients})



	
class Critic:
	def __init__(self, sess, feature_dim, action_dim, hidden_units, eta):
		self.sess = sess
		self.feature_dim = feature_dim
		self.action_dim = action_dim
		self.vars = None
		self.hidden_units = hidden_units
		self.buildNetwork()

		with tf.name_scope("Critic") as cope:
			y = tf.placeholder(tf.float32, [None, 1], "y")

		self.loss = tf.reduce_mean(tf.square(y - self.qa_critic))
		tf.scalar_summary("loss critic", self.loss)
		self.optimizer = tf.train.AdamOptimizer(eta)


	def buildNetwork(self)
		with tf.name_scope("Critic") as scope:
			self.s = tf.placeholder(tf.float32, [None, self.feature_dim], "s")
			self.a = tf.placeholder(tf.float32, [None, self.action_dim], "a")
			Ws_critic = tf.Variable(tf.truncated_normal([self.feature_dim, self.hidden_units]))
			Wa_critic = tf.Variable(tf.truncated_normal([self.action_dim, self.hidden_units]))
			b_critic = tf.Variable(tf.truncated_normal([self.hidden_units],0.5,1e-01))
		
			self.hidden_critic = tf.nn.relu(tf.matmul(s, Ws_critic) + tf.matmul(a, Wa_critic) + b_critic)
		#	self.hidden_critic = tf.tanh(tf.matmul(s, Ws_critic) + tf.matmul(a, Wa_critic) + b_critic)
			Wh_critic = tf.Variable(tf.truncated_normal([self.hidden_units, 1], 0, 1e-02))
			bh_critic = tf.Variable(tf.zeros([1]))
			
			self.qa_critic = tf.matmul(self.hidden_critic, Wh_critic) + bh_critic

	def evalCritic(self, s, a):
		return self.sess.run(self.qa_critic,
				feed_dict = {self.s: s, self.a: a})

	def train(self):
		self.sess.run
				critic.train(ss__, as__, ys__)

		
class ReplayBuffer:
	def __init__(self, size):
		self.size = size
		self.s__ = np.zeros([REPLAY_BUFFER_SIZE, FEATURE_DIM]).astype(np.float32)
		self.s_print__ = np.zeros([1, FEATURE_DIM]).astype(np.float32)

		self.a__ = np.zeros([REPLAY_BUFFER_SIZE, ACTION_DIM]).astype(np.float32)
		self.a_print__ = np.zeros([1, ACTION_DIM]).astype(np.float32)
		self.a_eval__ = np.zeros([1, ACTION_DIM]).astype(np.float32)

		self.r__ = np.zeros([REPLAY_BUFFER_SIZE])
		self.t__ = np.zeros([REPLAY_BUFFER_SIZE]).astype(np.bool)

		self.buff_i = 0
		self.buff_full = False
		self.seq = np.array(range(self.size))

	def setState(self, state):
		# Copy state into buffer
		self.s__[self.buff_i, :] = state

	def getCurrentState(self):
		return self.s__[np.newaxis, self.buff_i, :]

	def setAction(self, action):
		self.a__[self.buff_i, :] = action

	def getCurrentAction(self):
		returns self.a__[np.newaxis, self.buff_i, :]

	def setReward(self, reward):
		self.r__[self.buff_i] = reward

	def setTerminal(self, terminal):
		self.t__[self.buff_i] = terminal

	def advanceBuffer(self):
		self.buff_i = (self.buff_i + 1) % self.size
		if self.buff_i == 0:
			self.buff_full = True

	def addoneandwrap(x):
		return (x + 1) % REPLAY_BUFFER_SIZE

	def getSample(self):
		mask = np.ones(len(seq)).astype(np.bool)
		# don't train on the last state b/c the circular buffer will wrap around.
		mask[buffer] = False

		#randomly sample SAMPLE_SIZE samples from replay buffer
		sample_idx = random.sample(seq[mask], SAMPLE_SIZE)

		ss__ = s__[sample_idx, :]
		as__ = a__[sample_idx, :]
		rs__ = r__[sample_idx, np.newaxis]
		ts__ = t__[sample_idx, np.newaxis]

		# Find next state for samples. This will be buggy when the sampled index
		# is at edge of the circular buffer, but we don't care for now.
		ss_next__ = s__[np.apply_along_axis(addoneandwrap,0,sample_idx), :]

		return ss__, as__, rs__, ts__, ss_next__


class Chart:
	def updateActorChart


class Trainer:
	def __init__(self, sess, history_length, world):
		self.sess = sess
		self.history_length = history_length
		self.s_hist = np.zeros([self.history_length]).astype(np.float32)

		self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

		self.world = world

	def buildState(self, s, reset):
		# copy history back by one
		for i, v in enumerate(self.s_hist):
			if i >= self.history_length - 2:
				break
			s_hist[i+1] = v

		# First element contains latest x 
		self.s_hist[0] = s[0]
		# Latest xdot is unused
		# Last element contains target x
		self.s_hist[self.history_length - 1] = s[2]

		if reset:
			# if reset, reset full state of the history_length
			for i in range(self.history_length):
				self.s_hist[i] = s[0]

		self.replay_buffer.setState(self.s_hist)

	def updateChart(self, s_, a_):
		v = critic.evalCritic(self.replay_buffer.getCurrentState(), self.replay_buffer.getCurrentAction())

		updateActorChart(s_, a_, 0)
		updateActorChart(s_, v, 1)


	def train(self, actor, critic):
		self.sess.run(tf.initialize_all_variables())

		for step in range(1e10):
			s_ = self.g.getState()
			self.buildState(s_, False) 
			a_ = actor.actionActor(self.replay_buffer.getCurrentState())
			self.replay_buffer.setAction(a_)

			s_next_, r, terminal = self.world.step(a_)
			self.replay_buffer.setReward(r)
			self.replay_buffer.setTerminal(t)

			self.updateChart(s_, a_)

			self.replay_buffer.advanceBuffer()
			if self.replay_buffer.bufferFull():
				ss__, as__, rs__, ts__, ss_next__ = self.replay_buffer.getSample()
				aa = actor.actionActor(ss_next__)
				vv = critic.evalCritic(ss, aa)

				# Compute reward for each of the sample states
				ys__ = rs__ + GAMMA * vv
				# If next state is terminal, use reward only.
				ys__ = ts__ * rs__ + (1 - ts__) * ys__
				
				critic.train(ss__, as__, ys__)

				grad = critic.getActionGradient(ss__, as__, ys__)
				actor.train(ss__, grad)
	


