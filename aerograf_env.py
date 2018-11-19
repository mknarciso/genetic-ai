"""
Aerograf Environment Interface
"""

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
		return self.state, self.reward, self.game_over()

#	def render(self): 	
#		Not implemented

	def write_action(self):
		# Set action filename
		full_path = self.path+"bvr_action_"+str(self.frame)+".txt"
		# Lock semaphore
		open(self.path+"dqn.lock", 'a').close()
		# Write action
		with open(full_path,"w") as file:
			file.write(str(self.action))
		# Release semaphore
		os.remove(self.path+"dqn.lock")

	def read_state(self):
		# Set state filename
		full_path = self.path+"bvr_state_"+str(self.frame)+".txt"
		# Wait until file exists
		while not os.path.exists(full_path):
		    time.sleep(0.01)
		# Wait Aerograf release semaphore
		while os.path.exists(self.path+"aerograf.lock"):
		    time.sleep(0.01)
		# Read file
		if os.path.isfile(full_path):
			file_in = open(full_path, 'r')
			for i,y in enumerate(file_in.read().splitlines()):
				# First position is the reward
				if i==0:
					self.reward = float(y)
				# The others are the next state
				else:
					self.state[i-1] = float(y)
			file_in.close()
		# Destroy file after reading it
		os.remove(full_path)

	def game_over(self):
		# Check if episode is over by timeframe
		if self.frame >= MAX_FRAME-STEP:
			return True
		# Or by existence of gameover file
		full_path = self.path+"game_over.lock"
		if os.path.exists(full_path):
			os.remove(full_path)
			return True
		return False