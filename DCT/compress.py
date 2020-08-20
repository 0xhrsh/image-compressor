import cv2
from math import sqrt,cos
from pprint import pprint
import numpy as np 
from numpy import shape
import matplotlib.pyplot as plt 
import skimage
import time
from zizag import *
from huffman import HuffmanExample as compress

def dct(img):
	t1=time.time()
	Qa=np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])
	Qb=np.array([[17, 18, 24, 47, 99, 99, 99, 99],[18, 21, 26, 66, 99, 99, 99, 99],[24, 26, 56, 99, 99, 99, 99, 99],[47, 66, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99]])
	

	print("Enter Quality Factor: ")
	k=int(input())
	if(k>=50):
		s=200-2*k
		Qb=(50+(Qb*s))/100
		Qa=(100-k)/50
	else:
		s=5000/k
		Qa=50/k
		Qb=(50+(Qb*s))/100

	n=len(img)
	img=cv2.split(img)
	dimg=np.array(img)
	block_size=8
	t=dctMat(block_size)
	tt=t.transpose()
	for x in range(3):
		if(x!=0):
			QUANTIZATION_MAT=Qb
		else:
			QUANTIZATION_MAT=Qa
		for a in range(n//8):
			row_ind_1 = a*block_size                
			row_ind_2 = row_ind_1+block_size
			for b in range(n//8):

				# Findind DCT
				col_ind_1 = b*block_size                       
				col_ind_2 = col_ind_1+block_size
				block = img[x][ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]

				DCT = np.dot(t,np.dot(block,tt))

				DCT_normalized = np.divide(DCT,QUANTIZATION_MAT).astype(int)
				reordered = zigzag_single(DCT_normalized)
				reshaped= np.reshape(reordered, (block_size, block_size)) 
				img[x][row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshaped


				# uncomment to invert DCT

				#idct=np.multiply(DCT_normalized,QUANTIZATION_MAT)
				#idct=np.dot(np.dot(tt,idct),t)
				#dimg[x][row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = idct

				

				
	print("time of compression per frame: ",time.time()-t1)
	
	#Save File here
	fimg=np.array(img)
	for i in range(3):
		fimg[i]=np.array(img[i])
	fimg=cv2.merge(fimg)
	fimg=np.array(fimg.flatten())
	path=h.compress()

	#return np.uint8(cv2.merge(dimg))
	
def get_run_length_encoding(image):
    i = 0
    skip = 0
    stream = []    
    bitstream = ""
    image = image.astype(int)
    while i < image.shape[0]:
        if image[i] != 0:            
            stream.append((image[i],skip))
            bitstream = bitstream + str(image[i])+ " " +str(skip)+ " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1
    return bitstream


def dctMat(n):
	arr=np.zeros((n,n))
	for q in range(n):
		arr[0][q]=1/sqrt(n)
	for p in range(1,n):
		for q in range(n):
			arr[p][q]=sqrt(2/n)*cos(3.14*(2*q+1)*p/(2*n))
	return arr

start=time.time()
count=0
path=""  	# Path of input video
vidObj = cv2.VideoCapture(path) 
success=True
while success:
	success,img=vidObj.read()
	dct(img) # Frame 
	cv2.imwrite("frames/frame%d.jpg" %count,dct(img))
	count+=1
print(time.time()-start)


