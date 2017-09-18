# gridworld.py
#
# A very simple 2d environment with boundaries. An agent can move around in all directions
# with the goal of finding the goal state. 

class Action:
	up = 0
	down = 1
	left = 2
	right = 3

class GridWorld:

	def __init__(self, width, height, goal, hole):
		self.width = width 
		self.height = height
		self.goal = goal
		self.hole = hole
		self.reward = 1
		self.reset()

	def reset(self):
		self.state = (0,0)

	def getState(self):
		return self.state

	
	def step(self, action):
		x, y = self.state
		if action == Action.up:
			next_state = (x, (((y+1) < self.height) and y+1) or y)
		elif action == Action.down:
			next_state = (x, (((y-1) >= 0) and y-1) or 0)
		elif action == Action.right:
			next_state = ((((x+1) < self.width) and x+1) or x, y)
		elif action == Action.left:
			next_state = ((((x-1) >= 0) and x-1) or 0, y)
		else:
			print "invalid action %d " % action
 
		t = False
		r = 0
		if next_state == self.goal:
			r = self.reward
			t = True
		if next_state == self.hole:
			r = -self.reward
			t = True

		if t:
			self.reset()
		else:
			self.state = next_state
		
		return (self.state, r, t)
