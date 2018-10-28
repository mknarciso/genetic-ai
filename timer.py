import os
import time

iteration = 3
path = "C:/bvr_ai/"
start = time.clock()
while True:
	frame = 1400
	while True:

		print("at "+str(time.clock()))
		full_path = path+str(iteration)+"_bvr_state_"+str(frame)+".txt"
		while not os.path.exists(full_path):
		    time.sleep(1)
		print(full_path+" detected at "+str(time.clock()))
		full_path = path+str(iteration)+"_bvr_action_"+str(frame)+".txt"
		while not os.path.exists(full_path):
		    #time.sleep(1)
		    #if os.path.exists(path+"dqn.lock"):
		    print("locked")
		print(full_path+" detected at "+str(time.clock()))



		frame += 10
	interation += 1