"""
Aerograf DQN Interface
"""

## binding.pry
#  import code; code.interact(local=dict(globals(), **locals()))

import os
import numpy as np 		# Mathematical library
import time

MAX_FRAME = 4800
STEP = 10

class Aerograf(object):
	def __init__(self, path, state_space):
		self.iteration = 0
		self.frame = 0
		self.state = np.zeros(state_space)
		self.action = 0
		self.reward = 0
		self.path = path

	def reset(self, iteration):
		self.iteration = iteration
		self.frame = 0
		self.reward = 0
		self.read_state()
		
		return self.state

	def step(self, action):
		self.action = action
		#put action to aerograf
		self.write_action()
		#wait advance STEP frames (10frames ~ 250ms => human avg reaction)
		self.frame += STEP
		#get next state from aerograf
		self.read_state()

		#if self.frame%500==0:
		#	print("["+str(self.iteration)+"|"+str(self.frame)+"] ")
		#print(self.state)
		#print(self.action)
		#print(self.reward)
		
		return self.state, self.reward, self.game_over()

#	def render(self): 	
#		self.player.move()
#		self.enemy.move()

	def write_action(self):
		full_path = self.path+"bvr_action_"+str(self.frame)+".txt"
		#old_path = self.path+"bvr_action_"+str(self.frame-STEP)+".txt"
		#if os.path.exists(old_path):
		#	os.remove(old_path)
		open(self.path+"dqn.lock", 'a').close()
		with open(full_path,"w") as file:
			file.write(str(self.action))
		os.remove(self.path+"dqn.lock")

	def read_state(self):
		full_path = self.path+"bvr_state_"+str(self.frame)+".txt"
		while not os.path.exists(full_path):
		    time.sleep(0.01)
		while os.path.exists(self.path+"aerograf.lock"):
		    time.sleep(0.01)
		if os.path.isfile(full_path):
			file_in = open(full_path, 'r')
			for i,y in enumerate(file_in.read().splitlines()):
				if i==0:
					self.reward = float(y)
				else:
					self.state[i-1] = float(y)
			file_in.close()
		os.remove(full_path)

	def game_over(self):
		if self.frame >= MAX_FRAME-STEP:
			return True
		return False