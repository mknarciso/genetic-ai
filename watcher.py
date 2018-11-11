import numpy as np
import os
import time

times = 50
n_features = 6
memory_size = times*479
path = "C:/bvr_ai/"
memory = np.zeros((memory_size, n_features * 2 + 2))
reward = 0
_reward = 0
state = np.zeros(n_features)
_state = np.zeros(n_features)
action = 0
_action = 0
memory_counter = 0

for c in range(times):
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
					_reward = float(y)
				else:
					_state[i-1] = float(y)
			file_in.close()
			os.remove(state_path)

		if os.path.isfile(action_path):
			file_in = open(action_path, 'r')
			y = file_in.readline()
			_action = float(y)
			file_in.close()
			os.remove(action_path)

		if not f==0:
			transition = np.hstack((state, [action, _reward], _state))
			# replace the old memory with new memory
			index = memory_counter % memory_size
			memory[index, :] = transition
			memory_counter += 1
			print("[Frame: "+str(f-1)+"]"+str(transition))

		state = _state
		reward = _reward
		action = _action

		full_path = "C:/bvr_ai/game_over.lock"
		if os.path.exists(full_path):
			print("End iteraction "+str(c)+" at frame "+str(f))
			os.remove(full_path)
			break
	#import code; code.interact(local=dict(globals(), **locals()))
			
	print(c)

memory.tofile('C:/bvr_ai/memory/50_episodes_v10.dat')
print(memory)
