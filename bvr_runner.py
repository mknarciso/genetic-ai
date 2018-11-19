import os
import random
import numpy as np
import time

from aerograf_env import Aerograf
from DQN import DeepQNetwork

import matplotlib.pyplot as plt

episode_size = 479 

path = "C:/bvr_ai/"

# Get execution counter
data_counter = 0
if os.path.isfile(path+"counter.data"):
	with open(path+"counter.data", 'r') as file:
		data_counter = int(file.readline())
	with open(path+"counter.data","w") as file:
		file.write(str(data_counter+1))

# Make Dir for logs
this_path = path+"results/dqn_"+str(data_counter)+"/"
if not os.path.exists(this_path):
	os.makedirs(this_path)

# Config Variables
hidden_layers = "10"
strategy = "curriculumB"

# Start Environment
env = Aerograf(path,6)

# Start DQN
RL = DeepQNetwork(n_actions=4, 
				  n_features=6,
                  e_greedy=0.98,
            	  e_greedy_increment=0.00125,
                  memory_size=100*episode_size,
                  permanent_memory_size=50*episode_size,
                  training=False,
                  save_file=this_path+"nn/dqn",
                  mem_file='C:/bvr_ai/memory/50_episodes_v10.dat',
                  #import_file=path+"results/dqn_44/nn/dqn",
                  )

#Log initial configs
with open(this_path+"config.txt","w+") as f:
	f.write("hidden_layers: "+hidden_layers+"\n")
	f.write("n_actions: "+str(RL.n_actions)+"\n")
	f.write("n_features: "+str(RL.n_features)+"\n")
	f.write("lr: "+str(RL.lr)+"\n")
	f.write("gamma: "+str(RL.gamma)+"\n")
	f.write("epsilon_max: "+str(RL.epsilon_max)+"\n")
	f.write("replace_target_iter: "+str(RL.replace_target_iter)+"\n")
	f.write("memory_size: "+str(RL.memory_size)+"\n")
	f.write("permanent_memory_size: "+str(RL.permanent_memory_size)+"\n")
	f.write("batch_size: "+str(RL.batch_size)+"\n")
	f.write("epsilon_increment: "+str(RL.epsilon_increment)+"\n")
	f.write("epsilon: "+str(RL.epsilon)+"\n")
	f.write("ep_exp_decay: "+str(RL.exp_eps_decay)+"\n")
	f.write("training: "+str(RL.training)+"\n")
	f.write("save_file: "+str(RL.save_file)+"\n")
	f.write("strategy: "+str(strategy)+"\n")

# Init variables
score_history = []
lost_at = [0.]
avg_lvl = [0]
episode = 0
start_total = time.clock()
level_counter = np.zeros(9)
level_wins = np.zeros(9)
level = 0
acc_level = 0
acc_wins = 0

try:
	while True:
		step = 0
		# Waits for a new episode or the end of simulations
		while not os.path.exists(path+"bvr_state_0.txt") and not os.path.exists(path+"done.lock"):
			time.sleep(0.01)

		# Wait aerograf to end its simulations
		if os.path.exists(path+"done.lock"):
			time.sleep(1)
			os.remove(path+"done.lock")
			break

		# Initial state
		state = env.reset(episode)
		score = 0
		start = time.clock()
		level_counter[level] += 1

		#Main game loop
		while True:
			# Agent selects action based on State
			action = RL.choose_action(state)
			# Colects tuple from environment
			state_, reward, done = env.step(action)
			# Store tuple in memory
			RL.store_transition(state, action, reward, state_)

			# Train net each 24 steps
			if(step % 24 == 0):
				RL.learn()

			# t+1 becomes t
			state = state_

			#env.render()

			# Sums reward to Return Score
			score += reward

			if done:
				# Last learn
				RL.learn()
				end = time.clock()
				env.write_action()
				score_history.append(score)

				# Prints and logs results
				if strategy=="curriculumB":
					result = "[Ep: "+ str(episode) + "][Sc: " + str("%6.2f" % score) + "][Lv: "+ str(level) + "][Acc_wins: "+ str(acc_wins) + "]"+str(level_wins)+str(level_counter)+"[Spent: "+str("%6.2f" % (end-start))+"s]"
				else:
					result = "[Ep: "+ str(episode) + "][Sc: " + str("%6.2f" % score) + "][Lv: "+ str(level) + "][AvgLast5Lv: "+ str("%6.4f" % (sum(lost_at[-5:])/5.)) + "][AvgLv: "+ str("%6.4f" % (sum(lost_at)/float(len(lost_at)))) + "]"+str(level_wins)+str(level_counter)+"[Spent: "+str("%6.2f" % (end-start))+"s]"
				print(result)
				file = open(this_path+"log.txt","a+") 
				file.write(result+"\n")
				file.close()

				# Learning strategy to advance levels
				if strategy=="curriculumA":
					if score >= 1:
						level_wins[level] += 1
						level += 1
						if level > 8:
							level = 0
					else:
						lost_at.append(level)
						avg_lvl.append(sum(lost_at)/float(len(lost_at)))
						level = 0

				if strategy=="curriculumB":
					if acc_level > 8:
						if score >= 1:
							level_wins[level] += 1
							level += 1
							if level > 8:
								level = 0
						else:
							lost_at.append(level)
							avg_lvl.append(sum(lost_at)/float(len(lost_at)))
							level = 0
					else:
						if score>=1:
							acc_wins += 1;
							level_wins[level] += 1
						if acc_wins > 20:
							acc_wins = 0;
							level += 1;   
							acc_level += 1;

				if strategy=="sequential":
					if score >= 1:
						level_wins[level] += 1
					else:
						lost_at.append(level)
						avg_lvl.append(sum(lost_at)/float(len(lost_at)))
					level += 1
					if level > 8:
						level = 0
				break
			step += 1
		episode += 1
except KeyboardInterrupt:
	pass

# Time counter and final logs
end_total = time.clock()		
total = (end_total-start_total)

final_result = "[Total Time: "+str("%8.2f" % total)+"s]"+"[Avg Time: "+str("%6.2f" % (total/(episode+1))+"s]")
print(final_result)
with open(this_path+"log.txt","a") as file:
	file.write(final_result+"\n")

with open(this_path+"final_log.txt","w+") as f:
	f.write("Cost History:\n")
	f.write(str(RL.cost_his)+"\n\n")
	f.write("Score History:\n")
	f.write(str(score_history)+"\n\n")
	f.write("Lost at Level:\n")
	f.write(str(lost_at)+"\n\n")

print('game over')
