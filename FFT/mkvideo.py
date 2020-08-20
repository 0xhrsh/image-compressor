import cv2
import numpy as np
import glob
from moviepy.editor import *

img_array = []
for filename in glob.glob('frames/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

clip = VideoFileClip("project1.avi")

clip.write_gif("madeAGif.gif")