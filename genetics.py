import numpy as np

# Natural selection =D
def select(scores,survivors):
	#res = np.array(survivors)
	# import code; code.interact(local=dict(globals(), **locals()))
	amp = np.amax(scores)-np.amin(scores)
	points = (scores - np.amin(scores))/amp
	prob = points*points / np.sum(np.power(points,2))
	# import code; code.interact(local=dict(globals(), **locals()))
	nz = prob[prob != 0.]
	if amp<0.1:
		return np.random.choice(len(scores), survivors, replace=False)
	elif len(nz) >= survivors:
		return np.random.choice(len(scores), survivors, replace=False, p=prob)
	elif len(nz)>0:
		return np.random.choice(len(scores), survivors, replace=True, p=prob)
	else:
		return np.random.choice(len(scores), survivors, replace=False)

def breed(mates, survivors, species, mutation, dna_size):
	# import code; code.interact(local=dict(globals(), **locals()))
	# mates = np.split(dnas,survivors)
	new_generation = np.zeros((species,dna_size))
	i = 0
	for j in range(survivors):
		new_generation[j]=mates[j]
		i+=1
	for j in range(i,species):
		t = np.random.choice(len(mates), 2, replace=False)
		new_generation[j]=np.where(np.random.choice([True,False], dna_size),mates[t[0]] , mates[t[1]])
		new_generation[j]=mutate(new_generation[j],mutation,dna_size)
	return new_generation

def mutate(dna,mutation,dna_size):
	return np.where(np.random.choice([True,False], dna_size, p=[1-mutation, mutation]), dna , 2*np.random.random(dna_size)-1)

def select_dna(survivors,dna_size,dnas, scores):
	selected = select(scores,survivors)	
	print("Selected: " + str(selected))
	selected_dnas = np.zeros((survivors,dna_size))
	for i, mate_number in enumerate(selected):
		selected_dnas[i]=dnas[mate_number]
	return selected_dnas

def dyn_mutation(mutation,scores):
	if np.amax(scores)<0:
		dyn_mut = 0.4
	elif np.amax(scores)<1:
		dyn_mut = 0.3
	else:
		dyn_mut = mutation/np.amax(scores)
	return dyn_mut