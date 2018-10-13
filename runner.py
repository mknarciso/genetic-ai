import os
import random
import numpy as np 		# Mathematical library
import turtle 			# Graphics library
import autopilot as ap
import genetics as gen

from dogfight_env import Dogfight

## Constants

# Model
l = [14,9,4,3]
# l[0] = 14  # Inputs
# l[1] = 10
# l[2] = 5
# l[3] = 3 	 # Outputs
DNA_SIZE = (l[0]*l[1]+(l[1]+1)*l[2]+(l[2]+1)*l[3])

# Genetics
GENERATIONS = 15
SPECIES = 15
SURVIVORS = 4
MUTATION = 0.02

counter = 0
generation = 0
element = 0

## Set Params


## Initial DNA
dnas = 2*np.random.random((SPECIES,DNA_SIZE)) - 1 # zero mean
leg = np.loadtxt('save.txt', dtype=float)
dnas = leg
# leg = dnas
actual_best = np.zeros(DNA_SIZE)
#import code; code.interact(local=dict(globals(), **locals()))

env = Dogfight()

for generation in range(GENERATIONS):

	scores = np.zeros(SPECIES)

	for specie in range(SPECIES): 

		#Genetic properties 
		#leg = np.loadtxt('last_gen.txt', dtype=float)
		#dnas = leg
		# #Use saved pilot
		p1 = ap.Autopilot(dnas[specie],[l[0],l[1],l[2],l[3]])
		p2 = ap.Autopilot(leg[specie],[l[0],l[1],l[2],l[3]])

		blue_state, red_state = env.reset()	

		#Main game loop
		while True:
			blue_action = p1.fly_ai(blue_state)
			red_action = np.zeros(2)
			blue_state, red_state, blue_reward, done = env.step(blue_action, red_action)
			env.render()
			# # To show playable animation
			#game.show_status(player,enemy)
			# To show quick animation
			if counter%150==0:
				turtle.update()
			if done:
				break

		scores[specie] = blue_reward
		print("Final Score: " + str("%9.2f" % blue_reward) + " [GEN] "+ str(generation)+ " [#] "+ str(specie))
		
		env.reset()
		counter += 1
	#import code; code.interact(local=dict(globals(), **locals()))
	selected_dnas = gen.select_dna(SURVIVORS,DNA_SIZE,dnas, scores)
	dyn_mut = MUTATION #gen.dyn_mutation(MUTATION,scores)
	actual_best = dnas[np.argmax(scores)]
	dnas = gen.breed(selected_dnas, SURVIVORS, SPECIES, dyn_mut, DNA_SIZE)



a = dnas
np.savetxt('save.txt', a, fmt='%f')
print "Saved: " + str(dnas)
