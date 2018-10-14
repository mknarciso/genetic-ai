import os
import random
import numpy as np 		# Mathematical library
import turtle 			# Graphics library

from dogfight_env import Dogfight
from DQN_modified import DeepQNetwork

import matplotlib.pyplot as plt


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
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=1048576,
                  batch_size=50*700,
                  training=True,
                  import_file='saved/trained_dqn'
                  )

step = 0
score_history = []
for episode in range(600):
	blue_state, red_state = env.reset()
	score = 0
	#Main game loop
	while True:
		blue_action = RL.choose_action(blue_state)
		red_action = 0#RL.choose_action(red_state)

		blue_state_, red_state, blue_reward, done = env.step(
			translate_int_action(blue_action),
			translate_int_action(red_action)
			)

		RL.store_transition(blue_state, blue_action, blue_reward, blue_state_)

		if (step > 200) and (step % 50 == 0):
			RL.learn()

		blue_state = blue_state_

		env.render()

		score += blue_reward

		if episode%200==0:
			turtle.update()
		if done:
			score_history.append(score)
			print("Final Score: " + str("%9.2f" % score) + " [Episode] "+ str(episode))
			break
		step += 1
	# if episode%400==0:
	# 	plt.plot(np.arange(len(score_history)), score_history)
	# 	plt.ylabel('Score')
	# 	plt.xlabel('Episodes')
	# 	plt.show()

	# end of game

RL.plot_cost()

plt.plot(np.arange(len(score_history)), score_history)
plt.ylabel('Score')
plt.xlabel('Episodes')
plt.show()
x = np.array_split(score_history,30)
sums = []
for a in x:
	sums.append(sum(a))
plt.plot(np.arange(len(x)), sums)
plt.ylabel('Sum Scores each 50')
plt.xlabel('each 50 Episodes')
plt.show()

print('game over')
		
		
#import code; code.interact(local=dict(globals(), **locals()))

