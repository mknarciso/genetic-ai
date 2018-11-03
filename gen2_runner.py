import os
import random
import numpy as np 		# Mathematical library
import time

import autopilot_aer2 as ap
import genetics as gen

from aerograf_env import Aerograf
import matplotlib.pyplot as plt

## Constants

STATE_SPACE = 6
ACTION_SPACE = 4

# Model
l = [STATE_SPACE,8,ACTION_SPACE]
# l[0] = 14  # Inputs
# l[1] = 10
# l[2] = 5
# l[3] = 3 	 # Outputs
DNA_SIZE = (l[0]*l[1]+(l[1]+1)*l[2])

# Genetics
GENERATIONS = 15
SPECIES = 15
SURVIVORS = 4
MUTATION = 0.2

counter = 0
generation = 0
element = 0

## Set Params


## Initial DNA
#dnas = 2*np.random.random((SPECIES,DNA_SIZE)) - 1 # zero mean
dnas = np.loadtxt('genetic4_temp.txt', dtype=float)
actual_best = np.zeros(DNA_SIZE)
#import code; code.interact(local=dict(globals(), **locals()))

path = "C:/bvr_ai/"
env = Aerograf(path, STATE_SPACE)

gen_score = []

step = 0
score_history = []
episode = 0
start_total = time.clock()

for generation in range(GENERATIONS):

	scores = np.zeros(SPECIES)

	for specie in range(SPECIES): 
		alive = True
		level = 0
		while alive:
			# Waits for a new episode or for the end of simulations
			while not os.path.exists(path+"bvr_state_0.txt") and not os.path.exists(path+"done.lock"):
				time.sleep(0.01)
			# If aerograf ended its simulations
			if os.path.exists(path+"done.lock"):
				time.sleep(1)
				os.remove(path+"done.lock")
				break

			#Genetic properties 
			#dnas = np.loadtxt('last_gen.txt', dtype=float)
			# #Use saved pilot
			p1 = ap.Autopilot(dnas[specie],[l[0],l[1],l[2]], explorer_chance=0.005)

			state = env.reset(0)	
			actions_count = np.zeros(4)
			score = 0
			start = time.clock()

			#Main game loop
			while True:
				action = p1.fly_ai(state)
				state, reward, done = env.step(action)
				actions_count[action] += 1
				score += reward
				if done:
					if score > 100:
						level += 1
					else:
						alive = False
					end = time.clock()
					env.write_action()
					score_history.append(score)
					result = "[Level: "+ str(level) + "][Final Score: " + str("%9.2f" % score) + "][Spent: "+str("%6.2f" % (end-start))+"s]"+str(actions_count)
					print(result)
					file = open("C:/bvr_ai/logs/genetic_log5.txt","a") 
					file.write(result+"\n")
					file.close()
					break

			scores[specie] += score
		print("Final Score: " + str("%9.2f" % scores[specie])+ " [Level] " + str(level) + " [GEN] "+ str(generation)+ " [#] "+ str(specie)+"\n")
		
		#env.reset(0)
		counter += 1
	gen_score.append(sum(scores)/len(scores))

	#import code; code.interact(local=dict(globals(), **locals()))
	selected_dnas = gen.select_dna(SURVIVORS,DNA_SIZE,dnas, scores)
	dyn_mut = MUTATION #gen.dyn_mutation(MUTATION,scores)
	actual_best = dnas[np.argmax(scores)]
	dnas = gen.breed(selected_dnas, SURVIVORS, SPECIES, dyn_mut, DNA_SIZE)

	b = dnas
	np.savetxt('genetic5_temp.txt', b, fmt='%f')

end_total = time.clock()		
total = (end_total-start_total)

plt.plot(np.arange(len(gen_score)), gen_score)
plt.ylabel('Avg Score')
plt.xlabel('Generation')
plt.show()

a = dnas
np.savetxt('genetic5_final.txt', a, fmt='%f')
print("Saved: " + str(dnas))
