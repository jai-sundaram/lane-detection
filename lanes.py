import cv2
import numpy as np
import matplotlib.pyplot as plt
#getting the image
img = cv2.imread("road_image.jpg")

# #rendering the image
# cv2.imshow("img", img)
# cv2.waitKey(0)

#using canny edge detection
#edge detection is used to find boundaries of objects in  images
#we will be using it in this case to find regions in the image where there is a sharp change in intensity/colo
#remember that a pixel has a value between 0 and 255 - 0 is minmum intensity (completely black) and 255 is highest intensity (completely white)
#gradient is the change in brightness across pixels
#strong gradient is a strong change of brightness in pixels
#small gradient is a small change of birgness in pixels


#whereever there is a sharp change in intensity or brightness or strong gradient, there is a corresponding bright pixel in the gradient image
#when we trace through all of these bright pixels, we get edges

#we will be using to detect images in the image
#first converting the image to grayscale
#create a copy of the image
lane_copy = np.copy(img)
lane_gray = cv2.cvtColor(lane_copy, cv2.COLOR_RGB2GRAY)

#we need to now reduce the amount of noise in this image and smoothen it out
#to do so, we will be applying gaussian blurring
#remember that to smoothen out the image we are pretty much taking the averages of it and its neighboring pixels
#using gaussian  blurring with a 5x5 kernel (neighborhood size) with a deviation of 0,
lane_blur = cv2.GaussianBlur(lane_gray, (5,5), 0)
# cv2.imshow("blur", lane_blur)
# cv2.waitKey(0)

#now we are going to be using the canny method to identify edges in the image
#remember that an image is composed of pixels, so it can be read as a matrix (array of pixel intensities)

#we can also represent an image in a 2d coordinate space (x,y)
# x axis goes through the images width (number of columns)
# y axis goes through the images height (number of rows)
#preoduct of width and height gives you total number of pixels in the image
#images are a continous function of x and y
#since it is a math function, we can do math operations
#so we could determine the changes in brightnsss in the images (change in intensity)
#the canny function will take the derivative on (x,y) directions, and so it will measure the change in intensity
#a small derivative is a small change in intensity, big derivative is a big change
#this will help us overall compute the gradient in all directions in the image
#this is what the canny funciton does - it will compute the gradient in all directions in the image

#we will then trace the strongest gradient as a series of white pixels
#this canny function has a low threshold and high threshold
#if the gradient is higher than the upper threshold (high threshold), it is accepted as the edge pixel
#if it the gradient is below the lower threshold, then it is rejected (not part of the edge)
#if the gradient is in between the upper and lower threshold, it will be accepted as an edge pixel only if it is connected to a strong edge
#it is recommended to have a 1 to 3 ratio between lower and upper threshold btw
canny = cv2.Canny(lane_blur, 50, 150)
#keep in mind this canny function is what does the edge detection
#black background, with the edges being white
#again for now traces the most sharp changes in the intesnity


#now we want to pretty much isolate parts of the image and only look at the region that has the lane lines that we are trying to identify
#to do so, we will use matplotlib and plot the image
# plt.imshow(canny)
# plt.show()
#after doing so look at the x,y coordinates and select three points that will perfectly encapsulate the region we want if we are to construct a triangle connected by those three points
#this region is called the region of interest basically
#in this case the coordinates are (200,700) this is the left bottom of the image
#(1100, 700) which is the images right bottom
#(550, 250) this is the top of the image

#now let us create the triangle for this region of itnerest
#this will be a numpy array that contains the vertices
#however, since the fillPoly expects an array of multiple different polygons, we will need tosurrong these coordinates with another array, but this triangle will still technically be the only polygon
triangle = np.array([[(200, 700), (1100, 700), (500, 250)]])

#now we will opretty much have a mask (this black image) and we will fill it with this polygon (the triangle)
#creating the mask/black image
#pretty much creating an array of zeroes with the same shape as the canny image
#so this will be a completely black image (intensity of 0) with the same shape as the canny image)
mask = np.zeros_like(canny)
#now fill in the mask/black image with the triangle
#first paramemter is the mask, second is the polygon we are filling it with, third is the color which in this case is white
#this will return the mask with a triangle that is completely filled white that covers that region of interst
cv2.fillPoly(mask, triangle, 255)
cv2.imshow("mask", mask)
cv2.waitKey(0)

