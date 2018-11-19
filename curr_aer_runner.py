import os
import random
import numpy as np 		# Mathematical library
import time

from aerograf_env import Aerograf
from DQN_12 import DeepQNetwork

import matplotlib.pyplot as plt

episode_size = 479 

path = "C:/bvr_ai/"

data_counter = 0
if os.path.isfile(path+"counter.data"):
	with open(path+"counter.data", 'r') as file:
		data_counter = int(file.readline())
	with open(path+"counter.data","w") as file:
		file.write(str(data_counter+1))

this_path = path+"results/dqn_"+str(data_counter)+"/"
if not os.path.exists(this_path):
	os.makedirs(this_path)

hidden_layers = "10"

env = Aerograf(path,6)
RL = DeepQNetwork(n_actions=4, 
				  n_features=6,
                  #learning_rate=0.0008,
                  #reward_decay=0.995,
                  #e_greedy_start=0.9,
                  e_greedy=0.985,
            	  #e_greedy_increment=0.00125,
                  #replace_target_iter=50,
                  memory_size=100*episode_size,
                  permanent_memory_size=50*episode_size,
                  #batch_size=8*episode_size,
                  training=True,
                  save_file=this_path+"nn/dqn",
                  mem_file='C:/bvr_ai/memory/50_episodes_v10.dat',
                  import_file=path+"results/dqn_44/nn/dqn",
                  #e_exp_decay=0.0009,
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

			if(step % 24 == 0):
				RL.learn()

			state = state_
			actions_count[action] += 1

			#env.render()

			score += reward
			if done:
				RL.learn()
				end = time.clock()
				env.write_action()
				score_history.append(score)
				#result = "[Ep: "+ str(episode) + "][Sc: " + str("%6.2f" % score) + "][Lv: "+ str(level) + "][AvgLast5Lv: "+ str("%6.4f" % (sum(lost_at[-5:])/5.)) + "][AvgLv: "+ str("%6.4f" % (sum(lost_at)/float(len(lost_at)))) + "]"+str(level_wins)+str(level_counter)+str(actions_count)+"[Spent: "+str("%6.2f" % (end-start))+"s]"
				result = "[Ep: "+ str(episode) + "][Sc: " + str("%6.2f" % score) + "][Lv: "+ str(level) + "][Acc_wins: "+ str(acc_wins) + "]"+str(level_wins)+str(level_counter)+str(actions_count)+"[Spent: "+str("%6.2f" % (end-start))+"s]"
				print(result)
				file = open(this_path+"log.txt","a+") 
				file.write(result+"\n")
				file.close()
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
				break
			step += 1
		#import code; code.interact(local=dict(globals(), **locals()))
		episode += 1
except KeyboardInterrupt:
	pass

end_total = time.clock()		
total = (end_total-start_total)
			
#RL.plot_cost()

plt.plot(np.arange(len(RL.cost_his)), RL.cost_his)
plt.ylabel('Cost')
plt.xlabel('training steps')
plt.savefig(this_path+"loss.png")
plt.clf()

plt.plot(np.arange(len(score_history)), score_history)
plt.ylabel('Score')
plt.xlabel('Episodes')
plt.savefig(this_path+"raw_return.png")
plt.clf()

x = np.array_split(score_history,len(score_history)/20)
sums = []
for a in x:
	sums.append(sum(a))

plt.plot(np.arange(len(x)), sums)
plt.ylabel('Avg Scores')
plt.xlabel('each 20 Episodes')
plt.savefig(this_path+"avg_return.png")
plt.clf()

final_result = "[Total Time: "+str("%8.2f" % total)+"s]"+"[Avg Time: "+str("%6.2f" % (total/(episode+1))+"s]")
print(final_result)
with open(this_path+"log.txt","a") as file:
	file.write(final_result+"\n")


plt.plot(np.arange(len(lost_at)), lost_at)
plt.ylabel('Last Level')
plt.xlabel('Lives')
plt.savefig(this_path+"last_level.png")
plt.clf()

plt.plot(np.arange(len(avg_lvl)), avg_lvl)
plt.ylabel('Avg Level')
plt.xlabel('Lives')
plt.savefig(this_path+"avg_level.png")
plt.clf()

print('game over')
		
#import code; code.interact(local=dict(globals(), **locals()))

