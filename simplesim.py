# BVR Learning

## binding.pry
#  import code; code.interact(local=dict(globals(), **locals()))


import os
import random
import numpy as np 		# Mathematical library
import turtle 			# Graphics library

## Constants

# Model
L0 = 14  # Inputs
L1 = 10
L2 = 3 	 # Outputs

# Genetics
GENERATIONS = 20
SPECIES = 12
SURVIVORS = 3
MUTATION = 0.05
SEED = 0
# Tactical
FUEL = 570

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

# sigmoid function
def sig(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

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

def aa_score(blue, red):
	return abs(aspect_angle(red,blue))-abs(aspect_angle(blue,red))

def game_over(player):
	if player.fuel <= 0:
		return True
	return False

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
		
	def move(self):
		self.fd(self.speed)
		self.fuel -= 1
		
		#Boundary detection
		if self.xcor() > 290:
			self.setx(290)
			self.rt(60)
		
		if self.xcor() < -290:
			self.setx(-290)
			self.rt(60)
		
		if self.ycor() > 290:
			self.sety(290)
			self.rt(60)
		
		if self.ycor() < -290:
			self.sety(-290)
			self.rt(60)

	def is_desertor(self):
		
		#Boundary detection
		if self.xcor() > 290:
			return True
		
		if self.xcor() < -290:
			return True
		
		if self.ycor() > 290:
			return True
		
		if self.ycor() < -290:
			return True

		return False
				
class Player(Sprite):
	def __init__(self, spriteshape, color, startx, starty, heading):
		Sprite.__init__(self, spriteshape, color, startx, starty, heading)
		self.speed = 1
		self.fuel = FUEL

	def turn_left(self):
		self.lt(2)
		
	def turn_right(self):
		self.rt(2)

	def accelerate(self):
		self.speed += 0.5
		
	def decelerate(self):
		self.speed -= 0.5

class Game():
	def __init__(self):
		self.level = 1
		self.score = 0
		self.state = "playing"
		self.pen = turtle.Turtle()
		self.lives = 3
		
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
		if distance_score(blue,red) > 0.3:
			self.score += (aa_score(blue,red)/180)*(distance_score(blue,red))
		if blue.is_desertor():
			self.score -= 50
		
	def show_status(self, blue, red):
		self.pen.undo()
		msg = "[Fuel]%7d \n[Blue]%+7.2f | %.3f \n[ Red]%+7.2f \n[Advg]%+7.2f | %.3f \n[Scor]%+7.2f" % (blue.fuel, aspect_angle(blue,red), distance_score(blue,red), aspect_angle(red,blue), aa_score(blue,red), aa_score(blue,red)/180, self.score )
		self.pen.penup()
		self.pen.goto(-290,-290)
		self.pen.write(msg, font=("Courier", 16, "normal"))

#Autopilot
def aulopilot(me, enemy, game, dna0, dna1):
	p = np.zeros(14)
	# Ativacao
	p[0] = 1
	# Parametros
	p[1] = float(me.xcor())/300
	p[2] = float(me.ycor())/300
	p[3] = np.sin(me.heading()*np.pi/180)
	p[4] = np.cos(me.heading()*np.pi/180)
	p[5] = float(me.distance(enemy))/300
	p[6] = np.sin(me.towards(enemy)*np.pi/180)
	p[7] = np.cos(me.towards(enemy)*np.pi/180)
	p[8] = float(aspect_angle(me, enemy))/180
	p[9] = float(aspect_angle(enemy, me))/180
	p[10] = float(distance_score(me, enemy))
	p[11] = float(aa_score(me, enemy))/180
	p[12] = float(me.fuel)/2000
	p[13] = float(game.score)/2000
	
	l1 = sig(np.dot(p,dna0))
	l2 = sig(np.dot(l1,dna1))

	if l2[0]>l2[1] and l2[0]>l2[2]:
		me.turn_left()
		# print("Left")
	if l2[2]>l2[0] and l2[2]>l2[1]:
		me.turn_right()
		# print("Right")
	# print(l2)

# Natural selection =D
def select(scores,survivors):
	#res = np.array(survivors)
	# import code; code.interact(local=dict(globals(), **locals()))
	amp = np.amax(scores)-np.amin(scores)
	points = (scores - np.amin(scores))/amp
	prob = points / np.sum(points)
	if len(prob) < survivors:
		return np.random.choice(len(scores), survivors, replace=True, p=prob)
	elif len(prob)==0:
		return np.random.choice(len(scores), survivors, replace=False)
	else:
		return np.random.choice(len(scores), survivors, replace=False, p=prob)

def breed(mates, survivors, species, mutation):
	# import code; code.interact(local=dict(globals(), **locals()))
	# mates = np.split(dnas,survivors)
	dna_size = (L0*L1+L1*L2)
	new_generation = np.zeros((SPECIES,dna_size))
	i = 0
	for j in range(survivors):
		new_generation[j]=mates[j]
		i+=1
	for j in range(i,species):
		t = np.random.choice(len(mates), 2, replace=False)
		new_generation[j]=np.where(np.random.choice([True,False], dna_size),mates[t[0]] , mates[t[1]])
	return new_generation


counter = 0
generation = 0
element = 0

## Initial DNA
np.random.seed(SEED)
dna_size = (L0*L1+L1*L2)
dnas = 2*np.random.random((SPECIES,dna_size)) - 1 # zero mean

for generation in range(GENERATIONS):
	scores = np.zeros(SPECIES)
	for specie in range(SPECIES): 
		#Create game object
		game = Game()

		#Draw the game border
		game.draw_border()

		#Create my sprites
		player = Player("triangle", "blue", 0, -280, 90)
		enemy = Player("triangle", "red", 0, 280, 270)

		#Keyboard bindings
		turtle.onkey(enemy.turn_left, "Left")
		turtle.onkey(enemy.turn_right, "Right")
		turtle.onkey(enemy.accelerate, "Up")
		turtle.onkey(enemy.decelerate, "Down")
		turtle.listen()

		#Genetic properties 
		dna = dnas[specie]
		dna0 = np.reshape(dna[0:(L0*L1)], (L0,L1))
		dna1 = np.reshape(dna[(L0*L1):],(L1,L2))

		#Main game loop
		while not game_over(player):
			aulopilot(player,enemy,game,dna0,dna1)
			game.update_score(player,enemy)
			player.move()
			enemy.move()
			# # To show playable animation
			#game.show_status(player,enemy)
			# To show quick animation
			if counter%24==0:
				turtle.update()
		scores[specie] = game.score
		print("Final Score: " + str("%9.2f" % game.score) + " [GEN] "+ str(generation)+ " [#] "+ str(specie))
		
		turtle.reset()
		player.reset()
		enemy.reset()
		counter += 1
	# import code; code.interact(local=dict(globals(), **locals()))
	selected = select(scores,SURVIVORS)
	selected_dnas = np.zeros((SURVIVORS,dna_size))
	for i, mate_number in enumerate(selected):
		selected_dnas[i]=dnas[mate_number]
	dnas = breed(selected_dnas, SURVIVORS, SPECIES, MUTATION)
	print "Selected: " + str(selected)
	# delay = raw_input("Press enter to go. > ")

