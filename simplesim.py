# BVR Learning

## binding.pry
#  import code; code.interact(local=dict(globals(), **locals()))


import os
import random
import numpy as np 		# Mathematical library
import turtle 			# Graphics library
import autopilot as ap
import genetics as gen

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
SEED = 2
# Tactical
FUEL = 700

turtle.setup( width = 700, height = 700, startx = 0, starty = 0)
#Required by MacOSX to show the window
turtle.fd(0)
#Set the animations speed to the maximum
turtle.speed(10)
#Change the background color
turtle.bgcolor("black")
#Hide the default turtle
turtle.ht()
#This saves memory
turtle.setundobuffer(1)
#This speeds up drawing
# # To show playable animation
# turtle.tracer(1)
# To speedup evolution
turtle.tracer(0,0)

np.random.seed(SEED)



def aspect_angle(from_t, to_t):
	aa = float(to_t.heading() - from_t.towards(to_t))
	if aa < -180:
		aa += 360
	if aa > 180:
		aa -= 360
	return aa

def distance_score(blue, red):
	d = blue.distance(red)
	score = 0
	if d<50:
		score = d/50
	else:
		score = 50/d
	return score

def desertor_score(me):
	dist = float(me.distance(0,0))
	if dist > 280:
		return -dist/280
	return 0

def aa_score(blue, red):
	return abs(aspect_angle(red,blue))-abs(aspect_angle(blue,red))

def game_over(player):
	if player.fuel <= 0:
		return True
	return False

def manual_control(who):
	# turtle.onkey(enemy.update_state([1,0]), "Left")
	# turtle.onkey(enemy.update_state([0,1]), "Right")
	#Keyboard bindings
	turtle.onkey(who.turn_left, "Left")
	turtle.onkey(who.turn_right, "Right")
	turtle.onkey(who.accelerate, "Up")
	turtle.onkey(who.decelerate, "Down")
	turtle.listen()

class Sprite(turtle.Turtle):
	def __init__(self, spriteshape, color, startx, starty, heading):
		turtle.Turtle.__init__(self, shape = spriteshape)
		self.speed(0)
		self.penup()
		self.color(color)
		self.fd(0)
		self.goto(startx, starty)
		self.seth(heading)
		self.speed = 1
		
				
class Player(Sprite):
	def __init__(self, spriteshape, color, startx, starty, heading):
		Sprite.__init__(self, spriteshape, color, startx, starty, heading)
		self.speed = 1
		self.fuel = FUEL
		self.state = np.zeros(2)

	def update_state(self,state):
		self.state = state
		# print(self.state)

	def process_state(self):
		if self.state[0]==1:
			self.turn_left()
		if self.state[1]==1:
			self.turn_right()

	def turn_left(self):
		self.lt(2)
		
	def turn_right(self):
		self.rt(2)

	def accelerate(self):
		self.speed += 0.5
		
	def decelerate(self):
		self.speed -= 0.5

	def move(self):
		self.process_state()
		self.fd(self.speed)
		self.fuel -= 1

class Game():
	def __init__(self):
		self.score = 0
		self.pen = turtle.Turtle()
		
	def draw_border(self):
		#Draw border
		self.pen.speed(0)
		self.pen.color("white")
		self.pen.pensize(3)
		self.pen.penup()
		self.pen.goto(-300, 300)
		self.pen.pendown()
		for side in range(4):
			self.pen.fd(600)
			self.pen.rt(90)
		self.pen.penup()
		self.pen.ht()

	def update_score(self, blue, red):
		# target score
		radius = blue.distance(0,0)
		# if radius<100:
		# 	self.score += (100-radius)/600
		if distance_score(blue,red) > 0.3:
			self.score += (aa_score(blue,red)/180)*(distance_score(blue,red))
		self.score += desertor_score(blue)
		
	def show_status(self, blue, red):
		# self.pen.reset()
		self.pen.undo()
		msg = "[Fuel]%7d \n[Blue]%+7.2f | %.3f \n[ Red]%+7.2f \n[Advg]%+7.2f | %.3f \n[Scor]%+7.2f" % (blue.fuel, aspect_angle(blue,red), distance_score(blue,red), aspect_angle(red,blue), aa_score(blue,red), aa_score(blue,red)/180, self.score )
		self.pen.penup()
		self.pen.goto(-290,-290)
		self.pen.write(msg, font=("Courier", 16, "normal"))

def flight_params(me, enemy, game):
	p = np.zeros(14)
	# Ativacao
	p[0] = 1
	# Parametros
	p[1] = float(me.distance(0,0)/300) 			# Distance to objective
	p[2] = np.sin(me.towards(0,0)*np.pi/180)	# sin of angle to objective
	p[3] = np.cos(me.towards(0,0)*np.pi/180)	# cos of angle to objective
	p[4] = np.sin(me.heading()*np.pi/180)		# sin of my heading
	p[5] = np.cos(me.heading()*np.pi/180)		# cos of my heading
	p[6] = float(me.distance(enemy))/300		# distance towards the enemy
	p[7] = np.sin(me.towards(enemy)*np.pi/180)	# sin of heading to the enemy
	p[8] = np.cos(me.towards(enemy)*np.pi/180)	# cos of heading to the enemy
	p[9] = float(aspect_angle(me, enemy))/180	# aspect angle of the enemy
	p[10] = float(aspect_angle(enemy, me))/180	# my aspect angle to the enemy
	p[11] = float(distance_score(me, enemy))	# calculated distance score
	p[12] = float(aa_score(me, enemy))/180		# calculated aspect angle score
	p[12] = float(desertor_score(me))
	#p[13] = float(game.score)/2000				# total game score
	return p

counter = 0
generation = 0
element = 0

## Set Params


## Initial DNA
dnas = 2*np.random.random((SPECIES,DNA_SIZE)) - 1 # zero mean
#leg = np.loadtxt('save.txt', dtype=float)
#dnas = leg
leg = dnas
actual_best = np.zeros(DNA_SIZE)
#import code; code.interact(local=dict(globals(), **locals()))

for generation in range(GENERATIONS):

	scores = np.zeros(SPECIES)

	for specie in range(SPECIES): 

		## Initialize the Game itself
		#Create game object
		game = Game()
		#Draw the game border
		game.draw_border()
		#Create my sprites
		player = Player("triangle", "blue", 50*(2*np.random.random(1) - 1)[0], -280, 90)
		enemy = Player("triangle", "red", 50*(2*np.random.random(1) - 1)[0], 280, 270)

		#Genetic properties 
		#leg = np.loadtxt('last_gen.txt', dtype=float)
		#dnas = leg
		# #Use saved pilot
		p1 = ap.Autopilot(dnas[specie],[l[0],l[1],l[2],l[3]])
		p2 = ap.Autopilot(leg[specie],[l[0],l[1],l[2],l[3]])

		manual_control(enemy)

		#Main game loop
		while not game_over(player):
			# enemy.update_state(p1.fly_me(turtle))
			player.update_state(p1.fly_ai(flight_params(player,enemy,game)))
			#enemy.update_state(p2.fly_ai(flight_params(enemy,player,game))) # using saved for enemy
			game.update_score(player,enemy)
			player.move()
			enemy.move()
			# # To show playable animation
			#game.show_status(player,enemy)
			# To show quick animation
			if counter%15==0:
				turtle.update()
		scores[specie] = game.score
		print("Final Score: " + str("%9.2f" % game.score) + " [GEN] "+ str(generation)+ " [#] "+ str(specie))
		
		turtle.reset()
		player.reset()
		enemy.reset()
		counter += 1
	#import code; code.interact(local=dict(globals(), **locals()))
	selected_dnas = gen.select_dna(SURVIVORS,DNA_SIZE,dnas, scores)
	dyn_mut = MUTATION #gen.dyn_mutation(MUTATION,scores)
	actual_best = dnas[np.argmax(scores)]
	dnas = gen.breed(selected_dnas, SURVIVORS, SPECIES, dyn_mut, DNA_SIZE)



a = dnas
np.savetxt('save.txt', a, fmt='%f')
print("Saved: " + str(dnas))
