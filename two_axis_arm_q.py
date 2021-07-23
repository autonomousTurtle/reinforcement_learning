from typing import FrozenSet
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


im = Image.new("RGB", (512, 512), (128, 128, 128))
draw = ImageDraw.Draw(im)


def draw_circle(x,y,r,red=0,green=0,blue=255):
    leftUpPoint = (x-r, y-r)
    rightDownPoint = (x+r, y+r)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=(blue,green,red))
    return

def render_arm(theta1,theta2):
    # set up environment here
    base_x = im.height/2 # initial x, located in center
    base_y = im.width/2 # initial y, located in center
    r = 30 # the joint circle radius
    d1 = 50 # the arm segment length
    d2 = 40 # the arm distal arm segment length

    #distance formula
    #d = sqrt((x_2-x_1)^2 + (y_2-y_1)^2)

    x1 = d1*np.cos(theta1)
    y1 = d1*np.sin(theta1)
    print(x1,y1)
    #determine location of first circle

    x2 = x1+d2*np.cos(theta2)
    y2 = x2+d2*np.sin(theta2)

    print(x2,y2)


    #draw the center circle
    draw_circle(base_x,base_y,r)
    draw_circle(x1,y1,r, 50, 80, 90)
    draw_circle(x2,y2,r, 0, 255, 0)
    draw.line((0, im.height, im.width, 0), fill=(255,0,0), width=8)
    return




x1 = np.sin(np.pi)
y1 = 100

x2 = 200
y2 = 200

#angles between the 
theta1 = 0
theta2 = 0

render_arm(theta1, theta2)


#draw.rectangle((100,100,200,200), fill=(0,255,0))






#text
#font = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', 48)
#draw.multiline_text((0,0), 'Pillow example', fill=(0,0,0), font=FrozenSet)

cv2.imshow("image", np.array(im))  # show it!
cv2.waitKey(0)
