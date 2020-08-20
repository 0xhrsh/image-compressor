import cv2
from math import sqrt,cos
from pprint import pprint
import numpy as np 
from numpy import shape
import matplotlib.pyplot as plt 
import skimage
import time
from huffman import HuffmanExample as HuffmanCoding
from fftOptimized import *
import glob

trunt_wrap = 0
n=0

def fft(img): 
		t1 = time.time()
		img=cv2.split(img)
		dimg=np.array(img)
		for i in range(3):
			fft = FFT2(dimg[i])
			trunt = np.amax(abs(fft))*trunt_wrap
			np.place(fft,abs(fft)<trunt,complex(0))
			dimg[i] = INV_FFT2(fft)
		return np.uint8(cv2.merge(dimg))

start=time.time()
count=0
print("Enter trunk factor:")
trunt_wrap=float(input())
img_array=[]
path="giphy.gif"
vidObj = cv2.VideoCapture(path) 
success=True
if success:
	success,img=vidObj.read()
	n=len(img)
	preset(n)
	img=fft(img)
	img_array.append(img)
	cv2.imwrite("frames/frame%d.jpg" %count,img)
	count+=1
while success:
	success,img=vidObj.read()
	img=fft(img)
	img_array.append(img)
	cv2.imwrite("frames/frame%d.jpg" %count,img)
	# fimg=np.array(img.flatten())
	# bitstream=""
	# file=open("image/image%d.txt"%count,"w+")
	# for i in range(fimg.shape[0]):
	# 	file.write(str(fimg[i])+" ")
	# file.close()
	# h = HuffmanCoding("image/image%d.txt"%count)
	# output_path = h.compress()
	# print(output_path)
	count+=1
print(time.time()-start)

