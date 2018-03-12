# BVR Learning

import os
import random

#Import the Turtle module
import turtle
turtle.setup( width = 700, height = 700, startx = 0, starty = 0)
#Required by MacOSX to show the window
turtle.fd(0)
#Set the animations speed to the maximum
turtle.speed(0)
#Change the background color
turtle.bgcolor("black")
#Hide the default turtle
turtle.ht()
#This saves memory
turtle.setundobuffer(1)
#This speeds up drawing
turtle.tracer(1)



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
		self.speed = 0.3
		self.fuel = 30000

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
		# if blue.is_desertor:
		# 	self.score -= 50
		
	def show_status(self, blue, red):
		self.pen.undo()
		msg = "[Blue]%+7.2f | %.3f \n[ Red]%+7.2f \n[Advg]%+7.2f | %.3f \n[Scor]%+7.2f" % ( aspect_angle(blue,red), distance_score(blue,red), aspect_angle(red,blue), aa_score(blue,red), aa_score(blue,red)/180, self.score )
		self.pen.penup()
		self.pen.goto(-290,-290)
		self.pen.write(msg, font=("Courier", 16, "normal"))

#Create game object
game = Game()

#Draw the game border
game.draw_border()

#Create my sprites
player = Player("triangle", "blue", 0, -280, 90)
enemy = Player("triangle", "red", 0, 280, 270)

#Keyboard bindings
turtle.onkey(player.turn_left, "Left")
turtle.onkey(player.turn_right, "Right")
turtle.onkey(player.accelerate, "Up")
turtle.onkey(player.decelerate, "Down")
turtle.listen()

#Main game loop
while True:
	player.move()
	enemy.move()
	game.update_score(player,enemy)
	game.show_status(player,enemy)

delay = raw_input("Press enter to finish. > ")

