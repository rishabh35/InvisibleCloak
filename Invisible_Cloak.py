import numpy as np
import cv2
from collections import deque
def nothing(x):
	pass
cv2.namedWindow('Threshold')
cap = cv2.VideoCapture(0)
mask = None
count = 0
h, s, v = 100, 100, 100
cv2.createTrackbar('Hue       ', 'Threshold',0,179,nothing)
cv2.createTrackbar('Saturation', 'Threshold',0,255,nothing)
cv2.createTrackbar('Value     ', 'Threshold',0,255,nothing)
while(True):
	ret, frame = cap.read() 
	if ret==True:
		hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
		frame=cv2.flip(frame,1) 
		hsv=cv2.flip(hsv,1) 
		h=cv2.getTrackbarPos('Hue       ','Threshold')
		s=cv2.getTrackbarPos('Saturation','Threshold')
		v=cv2.getTrackbarPos('Value     ','Threshold')  
		lower=np.array([h-30,s,v], dtype=np.uint8)
		upper=np.array([h+30,255,255], dtype=np.uint8)
		mask=cv2.inRange(hsv, lower, upper)     
		mask=cv2.bitwise_and(hsv,hsv,mask=mask)     
		cv2.imshow("Threshold", mask)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	else:
		break
cv2.destroyAllWindows()

toMask = None
cv2.namedWindow('Background')
while(True):
	ret, frame=cap.read()
	if ret==True:
		frame=cv2.flip(frame,1)
		cv2.imshow("Background", frame)
		
		key = cv2.waitKey(1) & 0xFF
		if key==ord("s"):
			toMask = frame
		if key==ord("q"):
			break
	else:
		break
cv2.destroyAllWindows()

cv2.namedWindow('Frame')
while(True):
	ret, frame=cap.read()
	if ret==True:
		hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
		frame=cv2.flip(frame,1)
		hsv=cv2.flip(hsv,1) 
		  
		lower=np.array([h-30,s,v], dtype=np.uint8)
		upper=np.array([h+30,255,255], dtype=np.uint8)
		mask=cv2.inRange(hsv, lower, upper)     
		mask_inv = cv2.bitwise_not(mask)

		frameLessCloak=cv2.bitwise_and(frame,frame,mask=mask_inv)  

		cloak=cv2.bitwise_and(toMask,toMask,mask=mask)     

		cv2.imshow("Frame", frameLessCloak + cloak)
		key = cv2.waitKey(1) & 0xFF
		if key==ord("s"):
			count=count+1
			cv2.imwrite("Frame%d.jpg" % count, frame) 
		if key==ord("q"):
			break
	else:
		break
cap.release()
cv2.destroyAllWindows()