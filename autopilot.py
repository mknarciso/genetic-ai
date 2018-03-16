import numpy as np
import turtle 			# Graphics library


# sigmoid function
def sig(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

class Autopilot():
	def __init__(self):
		self.act = np.zeros(2)

	def act_reset(self):
		self.act = np.zeros(2)

	def act_pos(self,pos):
		self.act[pos] = 1

	def get_act(self):
		return self.act

	#Autopilot
	def fly_ai(self,p, dna0, dna1, dna2):
		act = self.act_reset()
		l1 = sig(np.dot(p,dna0))
		l2 = sig(np.dot(l1,dna1))
		l3 = sig(np.dot(l2,dna2))

		if l3[0]>l3[1] and l3[0]>l3[2]:
			self.act_pos(0)
			#me.turn_left()
			# print("Left")
		if l3[2]>l3[0] and l3[2]>l3[1]:
			self.act_pos(1)
			# me.turn_right()
			# print("Right")
		return self.get_act()

	def fly_me(self):
		act = self.act_reset()
		#Keyboard bindings
		turtle.onkey(self.act_pos(0), "Left")
		turtle.onkey(self.act_pos(1), "Right")
		# turtle.onkey(enemy.accelerate, "Up")
		# turtle.onkey(enemy.decelerate, "Down")
		turtle.listen()
		return self.get_act()
