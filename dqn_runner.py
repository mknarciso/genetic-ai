import os
import random
import numpy as np 		# Mathematical library
import turtle 			# Graphics library

from dogfight_env import Dogfight
from DQN_modified import DeepQNetwork



def translate_array_action(array_action):
	if array_action[0]==1:
		return 1
	if array_action[1]==1:
		return 2
	return 0

def translate_int_action(int_action):
	act = np.zeros(2)
	if int_action==1:
		act[0]=1
	if int_action==2:
		act[1]=1
	return act


env = Dogfight()
RL = DeepQNetwork(n_actions=3, 
				  n_features=14,
                  learning_rate=0.05,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=1048576,
                  batch_size=1024
                  )

step = 0
for episode in range(500):
	blue_state, red_state = env.reset()
	score = 0
	#Main game loop
	while True:
		blue_action = RL.choose_action(blue_state)
		red_action = np.zeros(2)

		blue_state_, _, blue_reward, done = env.step(
			translate_int_action(blue_action),
			red_action
			)

		RL.store_transition(blue_state, blue_action, blue_reward, blue_state_)

		if (step > 200) and (step % 200 == 0):
			RL.learn()

		blue_state = blue_state_

		env.render()

		score += blue_reward

		# if episode%2000==0:
		# 	turtle.update()
		if done:
			print("Final Score: " + str("%9.2f" % score) + " [Episode] "+ str(episode))
			break
		step += 1

	# end of game
RL.plot_cost()
print('game over')
		
		
#import code; code.interact(local=dict(globals(), **locals()))

