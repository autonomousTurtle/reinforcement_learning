import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import pickle
import matplotlib.pyplot as plt

from numpy import random

#Hyperparameters
SIZE = 200
RENDER_SIZE = 512
NM_EPISODES = 60000 # number of training episodes
MOVE_PENALTY = 1
GOAL_REWARD = 0
epsilon = 0.5 # level of randomness, changes over time
EPS_DECAY = 0.999 #every episode will be epsilon * EPS_DECAY
SHOW_EVERY = 3000 # how often to render the environment
TOLERANCE = 10

start_q_table = None #if we had a pickled Q table, we'll put its filename here

LEARNING_RATE = 0.1
DISCOUNT = 0.95

class Robot:
    def __init__(self):
        self.theta1 = np.random.uniform(low = 0,high = np.pi)
        self.theta2 = np.random.uniform(low = 0,high = np.pi)
        
    def __str__(self):
        return f"{self.theta1}, {self.theta2}"

    def action(self, choice):
        #actions that the arm can take are to rotate the arm +.1 rad or -.1 rad on each revolute joint
        if choice == 0:
            self.move(theta1=-.1, theta2=0) # do not move
        if choice == 1:
            self.move(theta1=.1, theta2=0) # move axis 1 +
        if choice == 2:
            self.move(theta1=0, theta2=.1) # move axis 1 -
        if choice == 3:
            self.move(theta1=0, theta2=-.1) #move axis 2 +
        #if choice == 4:
        #    self.move(theta1=0, theta2=-.1) # move axis 2 -
    
    def move(self, theta1=False, theta2=False):
        #move randomly if no value passed
        if not theta1:
            self.theta1 += (np.random.randint(-1, 1))/10 # move max abs(.1) rad
        else:
            self.theta1 += theta1
        if not theta2:
            self.theta2 += (np.random.randint(-1, 1))/10 # move max abs(.1) rad
        else:
            self.theta2 += theta2

class Goal:
    def __init__(self):
        self.x = np.random.randint(-70, 71)
        self.y = np.random.randint(-70,71)

    def __str__(self):
        return f"{self.x}, {self.y}"
        

def draw_environment(SIZE,x1,y1,x2,y2,goal_x, goal_y):

    def draw_circle(x,y,r,red=0,green=0,blue=255):
        leftUpPoint = (x-r, y-r)
        rightDownPoint = (x+r, y+r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, fill=(blue,green,red))
        return
    
    r = 10
    edge=10
    

    base_x = RENDER_SIZE / 2
    base_y = RENDER_SIZE / 2

    #normalize to the center of the rendering (0,0) is top left, +x is right, +y is down
    x1 = x1 + (RENDER_SIZE/2)
    y1 = y1 + (RENDER_SIZE/2)
    x2 = x2 + (RENDER_SIZE/2)
    y2 = y2 + (RENDER_SIZE/2)
    goal_x = goal_x+(RENDER_SIZE/2)
    goal_y = goal_y+(RENDER_SIZE/2)


    draw.rectangle((0, 0, RENDER_SIZE, RENDER_SIZE), (30,30,30)) # draw overtop of the image to clear it out

    #draw the center circle
    draw.line((base_x,base_y,x1,y1), fill=(0,0,0), width=6)
    draw.line((x1,y1,x2,y2), fill=(0,0,0), width=6)
    draw_circle(base_x,base_y,r)
    draw_circle(x1,y1,r, 50, 80, 90)
    draw_circle(x2,y2,r, 0, 255, 0)

    draw.rectangle((goal_x-edge,goal_y-edge,goal_x+edge,goal_y+edge), fill=(0,255,0)) # goal block

    #render and show the image
    
    cv2.imshow("image", np.array(im))
    #cv2.waitKey(0)
    cv2.waitKey(1)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #break


#angles are clockwise, 0 is directly to the right, in radians
#render_arm(theta1, theta2)

#the state is going to be the distance (x,y) to the goal
  # show 

def calculate_arm(theta1,theta2):
    
    # set up environment here
    # (0,0) in coordinate space is in the top left, +Y is down, +X is left to right across

    base_x = 0 # initial x, located in center
    base_y = 0 # initial y, located in center
    d1 = 60 # the arm segment length
    d2 = 60 # the arm distal arm segment length

    x1 = d1*np.cos(theta1)
    y1 = d1*np.sin(theta1)

    x2 = x1+d2*np.cos(theta2)
    y2 = y1+d2*np.sin(theta2)

    j1 = (x1,y1)
    eef = (x2,y2)

    return j1, eef


if start_q_table is None:
    print("q_table dos not exist")
    #initialize the q_table if there is not one that is passed
    #make the state size a uniform distribution for theta 1 and theta 2 from 0 to pi in increments of .1 rad (31 spaces)        
    #q_table is going to measure 
    #Make the size of the grid to the size of the rendering (512x512)
    #OBS_SPACE_SIZE = 
    q_table={}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            #for iii in range(-SIZE+1, SIZE):
            #    for iiii in range(-SIZE+1, SIZE):
            q_table[((i,ii))] = [np.random.uniform(-3,0) for i in range(4)]
    
else:
    #load in the saved q_table using pickle 
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

#print(q_table)

episode_rewards = [] # array from tracking episode rewards


# set up visualization base script
#im = Image.new("RGB", (SIZE, SIZE), (30, 30, 30))
im = Image.new("RGB",(512,512),(30,30,30))
draw = ImageDraw.Draw(im)

for episode in range(NM_EPISODES):
    #print("episode", episode)
    robot = Robot()
    goal = Goal()

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0 # track this single episode reward
    for i in range(300): # 1000 moves per episode

        j1, eef = calculate_arm(robot.theta1, robot.theta2) # grab current position of arm       
        obs = (int(eef[0]-goal.x),int(eef[1]-goal.y)) # obs is the difference between the goal and the eef of the robot
        print(obs)
        # decide which action to take or if it should move randomly
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs]) #take the best action
        else:
            action = np.random.randint(0,4) # take one random action

        robot.action(action)


        ### MAYBE? Make goal move? ###
        #goal.move(action)

        j1, eef = calculate_arm(robot.theta1, robot.theta2) # find where robot eef is

        ##### UPDATE THIS PART TO BUCKET THE STATE SPACE AND SEE IF WE ARE IN THE CORRECT BUCKET
        if int(eef[0]) < (goal.x + TOLERANCE) and int(eef[0]) > (goal.x - TOLERANCE) and int(eef[1]) < goal.y+ TOLERANCE and int(eef[1]) > goal.y- TOLERANCE:
            reward = GOAL_REWARD
        else: 
            reward = -MOVE_PENALTY

        # Now we know the reward, let's calc YO!

        new_obs = (int(eef[0])-goal.x, int(eef[1]) - goal.y) # observe immediatley
        max_future_q = np.max(q_table[new_obs]) # find the future q max reward
        current_q = q_table[obs][action] # 

        if reward == GOAL_REWARD:
            new_q = GOAL_REWARD
        else:
            #update the q learning backprop algo
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q #update the q table with the correct value
        
        if show:
            draw_environment(SIZE, j1[0], j1[1], eef[0], eef[1],goal.x, goal.y)

        episode_reward+= reward
        if reward == GOAL_REWARD:
            #print(episode)
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

print(q_table)

moving_average = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_average))], moving_average)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
        


