import numpy as np
import os
import time

times = 45
n_features = 6
memory_size = times*479
path = "C:/bvr_ai/"
reward = 0
memory_counter = 0
level = 0
level_counter = np.zeros(9)
level_wins = np.zeros(9)
for c in range(times):
	score = 0	
	level_counter[level] += 1		
	for f in range(480):
		frame = 10*f
		state_path = path+"bvr_state_"+str(frame)+".txt"
		action_path = path+"bvr_action_"+str(frame)+".txt"
		while not os.path.exists(state_path) and not os.path.exists(action_path):
			time.sleep(0.01)

		while os.path.exists(path+"aerograf.lock"):
		    time.sleep(0.01)

		if os.path.isfile(state_path):
			file_in = open(state_path, 'r')
			for i,y in enumerate(file_in.read().splitlines()):
				if i==0:
					reward = float(y)
			file_in.close()
			os.remove(state_path)

		if os.path.isfile(action_path):
			os.remove(action_path)

		score += reward

	if score >= 1:
		level_wins[level] += 1
	level += 1
	if level > 8:
		level = 0
	result = "[Ep: "+ str(c) + "][Sc: " + str("%6.2f" % score) + "][Lv: "+ str(level) + "]"+str(level_wins)+str(level_counter)
	print(result)
