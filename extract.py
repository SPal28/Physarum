#citing geeksforgeeks.com for vidoe extraction method 
import cv2 
import os 


vid = cv2.VideoCapture("C:/Users/subhangipal/Documents/Physarum/video/smooth_240104_1_MA062_01.mp4") 


try: 
	
	 
	if not os.path.exists('data'): 
		os.makedirs('data') 

 
except OSError: 
	print ('Error: Creating directory of data') 

 
currentframe = 0

while(True): 
	
	 
	ret,frame = vid.read() 

	if ret: 
		 
		name = './data/frame' + str(currentframe) + '.jpg'
		print ('Creating...' + name) 

		 
		cv2.imwrite(name, frame) 

		 
		currentframe += 1
	else: 
		break


vid.release() 
cv2.destroyAllWindows() 
