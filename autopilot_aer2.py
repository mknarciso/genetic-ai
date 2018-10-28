import numpy as np

# sigmoid function
def sig(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

class Autopilot():

	def __init__(self, dna, layers, explorer_chance=0.02):
		self.layers = layers
		self.dna = dna
		self.explorer_chance = explorer_chance

	def split_dna(self,pos):
		if pos==0:
			return np.reshape(self.dna[0:(self.layers[0]*self.layers[1])], (self.layers[0],self.layers[1]))
		else:
			start = 0
			for i in range(pos-1):
				start += (1+self.layers[i])*self.layers[i+1]
			end = start + (1+self.layers[pos])*self.layers[pos+1]
			return np.reshape(self.dna[start:end],(self.layers[pos]+1,self.layers[pos+1]))

	#Autopilot
	def fly_ai(self, p):
		l1 = sig(np.dot(p,self.split_dna(0)))
		l2 = sig(np.dot(np.concatenate((np.ones(1),l1),axis=0),self.split_dna(1)))
		#l3 = sig(np.dot(np.concatenate((np.ones(1),l2),axis=0),self.split_dna(2)))
		if np.random.random_sample() < self.explorer_chance:
			max = np.argmax(l2)
			l2[max]=0
			return np.argmax(l2)
		else:
			return np.argmax(l2)