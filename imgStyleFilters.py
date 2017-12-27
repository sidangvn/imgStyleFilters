
'''This mini group project was written by Si Dang, Karishma and Krishna in Dec - 2017
	The mini project uses for making basic image style filters as Instagram's images filters according to Zenva Academy
'''

import cv2
import numpy as np

#Passing function
def banana(val):
	pass

#store values. The values (matrices) below depend on the developers set it up
identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
sharpen_kernel = np.array([[0, -1, 0], [-1,5,-1], [0, -1, 0]])
gaussian_kernel1 = cv2.getGaussianKernel(3, 0)
gaussian_kernel2 = cv2.getGaussianKernel(5, 0)
box_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32) / 9

#kernel function included some kernels above
kernels = [identity_kernel, sharpen_kernel, gaussian_kernel1, gaussian_kernel2, box_kernel]

#image's name if it is in the the same folder as this code file or it's path
imageI = cv2.imread('picture.jpg')
imageF = imageI.copy() #make a copy from the original image

#convert color image to gray image
grayI = cv2.cvtColor(imageI, cv2.COLOR_BGR2GRAY) 
grayF = grayI.copy() #make a copy from the gray image above

cv2.namedWindow('Image Style Filters') #app's windows name

#below is image filter function set up
cv2.createTrackbar('contrast', 'Image Style Filters', 1, 100, banana)
cv2.createTrackbar('brightness', 'Image Style Filters', 50, 100, banana)
cv2.createTrackbar('filter', 'Image Style Filters', 0, len(kernels) -1, banana)
cv2.createTrackbar('grayscale', 'Image Style Filters', 0, 1, banana)

count = 1 #saved image number 

while True:
	grayscale = cv2.getTrackbarPos('grayscale', 'Image Style Filters') #gray trackbar
	if grayscale == 0:
		cv2.imshow('Image Style Filters', imageF) #if the image is not gray now
	else:
		cv2.imshow('Image Style Filters', grayF) #if the image is gray now

	button = cv2.waitKey(1) & 0xFF #buttons for save or quit image
	if button == ord('q'): #press q button to close
		break
	elif button == ord('s'): #press s button to save the image
		if grayscale == 0:
			cv2.imwrite('Image Style Filters %d.jpg' % count, imageF) #if image is not gray
		else:
			cv2.imwrite('Image Style Filters %d.jpg' % count, grayF) #if image is gray now
		count = count + 1 #update count above
		
    #moving the bar to edit image or get the position for the trackbar
	contrast = cv2.getTrackbarPos('contrast', 'Image Style Filters') #contrast trackbar
	brightness = cv2.getTrackbarPos('brightness', 'Image Style Filters') #brightness trackbar
	kernel = cv2.getTrackbarPos('filter', 'Image Style Filters') #filter trackbar
	
    #kernel set up
	imageF = cv2.filter2D(imageI, -1, kernels[kernel])
	grayF = cv2.filter2D(grayI, -1, kernels[kernel])
	
    #edited image brightness and contrast
	imageF = cv2.addWeighted(imageF, contrast, np.zeros(imageF.shape, dtype = imageF.dtype), 0, brightness - 50)
    #image edited with gray with contrast and brightness
	grayF = cv2.addWeighted(grayF, contrast, np.zeros(grayI.shape, dtype = grayI.dtype), 0, brightness - 50)
	
cv2.destroyAllWindows() #close windows


