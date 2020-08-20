import cv2
import numpy as np
import math
org=cv2.imread("uncompressed.bmp")
dec=cv2.imread("decompressed.bmp")
org=np.array(cv2.split(org))
dec=np.array(cv2.split(dec))
shp=org.shape
#print(shp[0],shp[1]*shp[2])
error=0
for i in range(shp[0]):
	for j in range(shp[1]):
		for k in range(shp[2]):
			x=(((org[i][j][k]-dec[i][j][k])**2)/(shp[1]*shp[2]))
			#print(x)
			error+=x
print(math.log(error)/math.log(10))
