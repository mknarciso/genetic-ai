import os
import random
import numpy as np 		# Mathematical library
import time

from aerograf_env import Aerograf
from DQN_modified7 import DeepQNetwork

import matplotlib.pyplot as plt
path = "C:/bvr_ai/"
env = Aerograf(path,6)
RL = DeepQNetwork(n_actions=4, 
				  n_features=6,
                  learning_rate=0.001,
                  reward_decay=0.99,
                  e_greedy=0.98,
            	  #e_greedy_increment=0.001,
                  replace_target_iter=50,
                  memory_size=1000*480, #480000
                  batch_size=16*480,  #7680
                  training=True,
                  save_file=path+"nn/dqn_with_memory_train",
                  mem_file='C:/bvr_ai/memory/dqn7.dat',
                  #import_file=path+"nn/dqn7",
                  )

while True:
	step = 0
	score_history = []
	episode = 0
	start_total = time.clock()
	level_counter = np.zeros(9)
	level = 0
	try:
		while True:
			# Waits for a new episode or for the end of simulations
			while not os.path.exists(path+"bvr_state_0.txt") and not os.path.exists(path+"done.lock"):
				time.sleep(0.01)
			# If aerograf ended its simulations
			if os.path.exists(path+"done.lock"):
				time.sleep(1)
				os.remove(path+"done.lock")
				break
			state = env.reset(episode)
			actions_count = np.zeros(4)
			score = 0
			start = time.clock()
			level_counter[level] += 1
			#Main game loop
			while True:
				action = RL.choose_action(state)
				state_, reward, done = env.step(action)
				RL.store_transition(state, action, reward, state_)

				if(step % 48 == 0):
					RL.learn()

				state = state_
				actions_count[action] += 1

				#env.render()

				score += reward
				if done:
					end = time.clock()
					env.write_action()
					score_history.append(score)
					result = "[Ep: "+ str(episode) + "][Sc: " + str("%6.2f" % score) + "][Lv: "+ str(level) + "]"+str(actions_count)+str(level_counter)+"[Spent: "+str("%6.2f" % (end-start))+"s]"
					print(result)
					file = open("C:/bvr_ai/logs/log.txt","a") 
					file.write(result+"\n")
					file.close()
					if score > 100:
						level += 1
						if level > 8:
							level = 0
					else:
						level = 0
					break
				step += 1
			episode += 1
	except KeyboardInterrupt:
		pass
	end_total = time.clock()		
	total = (end_total-start_total)
				
	RL.plot_cost()

	plt.plot(np.arange(len(score_history)), score_history)
	plt.ylabel('Score')
	plt.xlabel('Episodes')
	plt.show()

	x = np.array_split(score_history,len(score_history)/20)
	sums = []
	for a in x:
		sums.append(sum(a))
	plt.plot(np.arange(len(x)), sums)
	plt.ylabel('Avg Scores')
	plt.xlabel('each 20 Episodes')
	plt.show()

	print("[Total Time: "+str("%8.2f" % total)+"s]"+"[Avg Time: "+str("%6.2f" % (total/(episode+1))+"s]"))
	print('game over')
	time.sleep(1)
		
#import code; code.interact(local=dict(globals(), **locals()))

