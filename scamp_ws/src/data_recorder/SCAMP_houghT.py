#!/usr/bin/env python
import numpy as np
import cv2
import tensorflow as tf
import re
import os
import glob


folder = "/home/andrew/turtlebot3_ws/src/Converter/src/Test2/data"
assert folder, "Provide the dataset folder"
experiment = glob.glob(folder + "/*")

outputFolder = "/home/andrew/turtlebot3_ws/src/Converter/src/houghT/"
grad=0.0

for exp in experiment:
	outp = outputFolder
	print(exp)
	images = [os.path.basename(x) for x in glob.glob(exp + "/img/*.jpeg")]
	for im in images:
		stamp = str(re.sub(r'\.jpeg$','',im))
		print(stamp)
		im = exp + "/img/"+ im
		im = cv2.imread(im)

		edges = cv2.Canny(im, 50, 150, apertureSize=3) 
		lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
		
		num=0

		for line in lines:
			
			num=num+1

			gradl=line.flatten()
				
			#gradl has 4 elements: [x_start,y_start,x_end,y_end]
				
			if gradl[2]-gradl[0]==0:
				grad="inf"
			else:
				grad=-(float(gradl[3])-float(gradl[1]))/(float(gradl[2])-float(gradl[0]))
				# negative at front cuz top-left origin
			
			#print("gradient is: ",grad)
			
			
    			x1,y1,x2,y2 = line[0]


			org=(((x1+x2)/2)-4,(y1+y2)/2)
			font = cv2.FONT_HERSHEY_SIMPLEX
			fontScale = 0.5
			color = (255, 0, 0)
			thickness = 2
			label="L"+str(num)
			
    			cv2.line(im,(x1,y1),(x2,y2),(0,255,0),2)
			cv2.putText(im,label,org,font,fontScale,color,thickness,bottomLeftOrigin=False)
			print("Line "+str(num)+" slop:   ",grad)


					
		
		addr = outputFolder + str(stamp)+".jpeg"
		cv2.imwrite(addr,im)
		
		
		

			


