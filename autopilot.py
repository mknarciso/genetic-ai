import numpy as np


# sigmoid function
def sig(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

class Autopilot():

	def __init__(self, dna, layers):
		self.layers = layers
		self.dna = dna

	def split_dna(self,pos):
		if pos==0:
			return np.reshape(self.dna[0:(self.layers[0]*self.layers[1])], (self.layers[0],self.layers[1]))
		else:
			start = 0
			for i in range(pos-1):
				start += self.layers[i]*self.layers[i+1]
			end = start + self.layers[pos]*self.layers[pos+1]
			return np.reshape(self.dna[start:end],(self.layers[pos],self.layers[pos+1]))

	#Autopilot
	def fly_ai(self, p):
		act = np.zeros(2)
		l1 = sig(np.dot(p,self.split_dna(0)))
		l2 = sig(np.dot(l1,self.split_dna(1)))
		l3 = sig(np.dot(l2,self.split_dna(2)))

		if l3[0]>l3[1] and l3[0]>l3[2]:
			act[0]=1
			#me.turn_left()
			# print("Left")
		if l3[2]>l3[0] and l3[2]>l3[1]:
			act[1]=1
			# me.turn_right()
			# print("Right")
		return act
