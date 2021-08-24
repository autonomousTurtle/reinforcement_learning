import numpy as np
import matplotlib.pyplot as plt 
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque

#script to animate a 3D robot model using forward kinematics

# PARAMETERS
L1 = 1
L2 = 1
L = L1 + L2
M1 = 1.0
M2 = 1.0
STEPS = 500
DT = 0.02
TIME_STOP = STEPS * DT
RENDER = True


def calculate_arm(state):
    
    # set up environment here

    base_x = 0 # initial x, located in center
    base_y = 0 # initial y, located in center
    d1 = 60 # the arm segment length
    d2 = 60 # the arm distal arm segment length

    x1 = L1*np.cos(state[0])
    y1 = L1*np.sin(state[0])

    x2 = x1+L2*np.cos(state[1])
    y2 = y1+L2*np.sin(state[1])

    return x1,y1,x2,y2

def animate(i):

    x = [0, path[:, 0][i], path[:, 2][i]]
    y = [0, path[:, 1][i], path[:, 3][i]]

    if i == 0:
        history_x.clear
        history_y.clear

    history_x.appendleft(x[2])
    history_y.appendleft(y[2])

    line.set_data(x,y)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*DT))
    return line, trace, time_text


############# Main program #############

# initialize th1 and th2 are angles randomly
th1 = np.radians(np.random.randint(0,360))
th2 = np.radians(np.random.randint(0,360))
state = [th1, th2]

path=[] #memory for the animation of where the robot went

for step in range(0,STEPS):

    #move the robot
    state[0]+=0.05
    state[1]-=0.025

    x1,y1,x2,y2 = calculate_arm(state)
    if RENDER: 
        path.append([x1,y1,x2,y2])
    
    #find the new state
    

    #give reward


path=np.array(path) #convert to numpy array


# create the plot to animate
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(autoscale_on=False, xlim = (-L, L), ylim=(-L, L))
ax.set_aspect('equal')
ax.grid()

line , = ax.plot([],[], 'o-', lw=2)
trace, = ax.plot([],[], ',-', lw=1)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=STEPS), deque(maxlen=STEPS)


#animate the dang thing
ani = animation.FuncAnimation(fig, animate, len(path), interval=DT*1000, blit = True)
plt.show()
