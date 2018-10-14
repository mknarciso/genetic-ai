"""
Dog fight simple emulator
"""

## binding.pry
#  import code; code.interact(local=dict(globals(), **locals()))


import os
import random
import numpy as np 		# Mathematical library
import turtle 			# Graphics library

## Constants
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


class Dogfight(object):
	def __init__(self):
		## Initialize the Game itself
		#Create game object
		self.game = self.Game()
		#Draw the game border
		self.game.draw_border()
		self._build_game()

	def _build_game(self):
		#Create my sprites
		self.player = self.Player("triangle", "blue", 50*(2*np.random.random(1) - 1)[0], -250, 90)
		self.enemy = self.Player("triangle", "red", 50*(2*np.random.random(1) - 1)[0], 250, 270)

	def reset(self):
		turtle.clear()
		self.game.reset()
		self.player.reset()
		self.enemy.reset()
		self._build_game()
		blue_state = self.flight_params( self.player, self.enemy, self.game)
		red_state = self.flight_params(self.enemy,self.player,self.game)
		return blue_state, red_state

	def step(self, blue_action, red_action):
		self.player.update_state(blue_action)
		self.enemy.update_state(red_action) # using saved for enemy
		self.game.update_score(self.player,self.enemy)
		blue_state = self.flight_params(self.player,self.enemy,self.game)
		red_state =  self.flight_params(self.enemy,self.player,self.game)
		blue_reward = self.game.step_score(self.player, self.enemy)
		done = game_over(self.player)
		return blue_state, red_state, blue_reward, done


	def render(self): 	
		self.player.move()
		self.enemy.move()
		# self.game.show_status(self.player,self.enemy)


	def flight_params(self, me, enemy, game):
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
								
	class Player(turtle.Turtle):
		def __init__(self, spriteshape, color, startx, starty, heading):
			turtle.Turtle.__init__(self, shape = spriteshape)
			self.speed(0)
			self.penup()
			self.color(color)
			self.fd(0)
			self.goto(startx, starty)
			self.seth(heading)
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

		def reset(self):
			self.score = 0

		def step_score(self, blue, red):

			ans = 0 
			# target score
			radius = blue.distance(0,0)
			# if radius<100:
			# 	self.score += (100-radius)/600
			if distance_score(blue,red) > 0.3:
				ans += (aa_score(blue,red)/180)*(distance_score(blue,red))
			ans += desertor_score(blue)
			# if desertor_score(blue) < 0:
			# 	ans = desertor_score(blue)
			# elif distance_score(blue,red) > 0.3:
			# 	if aa_score(blue,red) > 0.1:
			# 		ans = aa_score(blue,red)
			# 	elif aa_score(blue,red) < -0.1:
			# 		ans = aa_score(blue,red)
			return ans

		def update_score(self, blue, red):
			self.score += self.step_score(blue, red)

		def show_status(self, blue, red):
			# self.pen.reset()
			self.pen.undo()
			msg = "[Fuel]%7d \n[Blue]%+7.2f | %.3f \n[ Red]%+7.2f \n[Advg]%+7.2f | %.3f \n[Scor]%+7.2f" % (blue.fuel, aspect_angle(blue,red), distance_score(blue,red), aspect_angle(red,blue), aa_score(blue,red), aa_score(blue,red)/180, self.score )
			self.pen.penup()
			self.pen.goto(-290,-290)
			self.pen.write(msg, font=("Courier", 16, "normal"))

# def show_status(self, blue, red):
# 	# self.pen.reset()
# 	self.pen.undo()
# 	msg = "[Blue]%+7.2f | %.3f \n[ Red]%+7.2f \n[Advg]%+7.2f | %.3f \n[Scor]%+7.2f" % (aspect_angle(blue,red), distance_score(blue,red), aspect_angle(red,blue), aa_score(blue,red), aa_score(blue,red)/180, self.score )
# 	self.pen.penup()
# 	self.pen.goto(-290,-290)
# 	self.pen.write(msg, font=("Courier", 16, "normal"))

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
		# return 1
		score = d/50
	else:
		# return 0
		score = 50/d
	return score

def desertor_score(me):
	dist = float(me.distance(0,0))
	if dist > 280:
		return -(dist-280)/280
		# return -1
	return 0

def aa_score(blue, red):
	return abs(aspect_angle(red,blue))-abs(aspect_angle(blue,red))

def game_over(player):
	if player.fuel <= 0:
		return True
	return False