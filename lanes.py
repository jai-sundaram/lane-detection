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
lane_copy2 = np.copy(img)
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
# cv2.imshow("mask", mask)
# cv2.waitKey(0)

#remember that this is a black and white image and so it can be summarized using binary numbers
#the black regions consist of 0s and the white regions consist of 1s

#now we are going to apply this mask on the canny image so we only show the regionn of interest

#we pretty much combine these images by using the bitwise & operator
#it occurs between the two images, between the two arrays of pixels
#these two images have the same array shape, so the same dimensions, and the same amount of pixels
#now what the bitwise & operator does, is if at a particular location, when comparing two binary numbers, if there is a 0 at all in one of the numbers at that location, it will be 0 in the new number, otherwise if both numbers are 1, then it will be 1
#for example
# 110011
# 100001
# The result will be:
# 10001
#so now when we compare the two images and do the bitwise & operator, if there is a black pixel at a given location then the new pixel at that new location will be black
# but if they are both white it will be white
#now when we do that we will get that black mask image with a white outline of the region of interest

#now implement this and compute the & operator between the canny image and the mask image
roi = cv2.bitwise_and(canny, mask)
# cv2.imshow("roi", roi)
# cv2.waitKey(0)

#now we are going to use Hough Transform to identify the straight lines in the image (helps us identify lane lines)

#in hough transform we first draw a 2D coordinate plane of x and y and inside of it a straight line
# we know that a certain line can be written in form y= mx + b
#instead of writing this as a function of x and y we can represent it in parameteric space called hough space
#in this space (hough space), it will be (m,b)
#so now in this case the slope is the x coordinate and the y intercept is the y coordinate
#so now that entire line can be represented as a coordinate
#now in the original (x,y) coordinates we could have many different lines that pass through a certain coordinate, eaching have a different y intercept and slope
#as a result, all these different lines will produce different different coordinates in the hough space (coordinates will be the combinations of different (m,b) values)
#these coordinates will be connected using a line

#in the x,y coordinate space lets say we had two coordinates
# to find a line that would pass through both coordinates, heres what we do
#we try out different lines that would pass through ONE of the coordinates on the (x,y) plane
#now we plot each of these different lines in the hough space as coordinates and then connect through them with a line

#we will repeat this same process for the other coordinate on the (x,y) plane

#the point of intersection for these 2 lines in the hough space at a given point - this point is the line that will go through both cordinates in the (x,y) space

#again this same concept can be applied when we have more than one coordinate in the (x,y) space and looking for a line that runs througb all of them

#this is the logic that we use to find the solid lines in the gradient image
#the gradient image is again the series of white points that represent the edges in the image space
#when we look at like a series of points we can see that they kinda form/belong to a certain line

#now to do this it gets a bit more complicated
#instead of the hough space being a simple (b,m) plane, it will instead be a graph with multiple "bins" which pretty much lines go through and there will be multiple points of intesrsection in the hough space
#we need to look for the bin that has the greatest number of intersections
#the bins (b,m) coordinates will be taken, and the line with this y intercept and slope in the (X,y) space will be the line of best fit

#however, the cartesian plane (with the x,y) coordinates cant handle it when the line is simlpy a Vertical line and therefore the slope is infinity
#as a result we use polar coordinates
#the line is: p = xcos(theta) + ysin(theta)
#p is the perpendicular distance from the origin to that line, theta is the angle of inclination of the normal line from the x axis, measured in radians
#now in the hough space, they will be measured using p (rho)  and theta
#the coordinate will be (theta, rho)

#when plotting these lines as coordinates in hough space, we get a sinusoidal curve
#other than this small aspect, it is exactly the same
#more intersections in the hough transform means that that line crosses more points
#also one intersection means one vote
#for ech intersection there will be one vote given
#again the hough space will be in the form multiple grids and the bin with the most intersections represents the line of best fit

#now this is what will help us find the line that best defines the edge points in the gradient image

#now implementing it
#detect the lines with the masked image with that region of interest
#2nd and third parameters define resolution of the hough accumulator array (this is that grid which is pretty much a 2d array of rows and columns which contain the bins)
#2nd and third columns define the size of these bins
#rho is the distance resolution of the accumulator (in pixels)
#theta is the angle resolutions of the accumulator (in radians)
#the larger the bins, the less precision with which the lines are detected
#the smaller the rho and theta values, the smaller the bins, the higher the precision with which we detect the lines
#we dont want tot make the bins too small since it will take longer to run and still be innacurate
# in this case the theta is 1 degree
#pi is 180 degrees, so 1 degrees is pi/180

#to detemrine which bins to choose when drawing a line, and the optimal number of votes where a line corrresponding to that bin is drawn, we use thresholds
#thresholds is the minuimum nnumber of intersections/votes needed to accept a candidate line

#in this case the threshold of 100
#so the number of intersections in hough space in a bin has to be 100 for it to be accepted as a line
#the fifth argument is just a simple palceholder empty array
#the sixth argument is the minimum line length - min length of a line in pixels for it to be accepted
#the seventh argument is the maximum line gap - maximum distance in pixels between segmented lines which can be connected into a single line
lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

#now we want to display these lines on the real/original image
#first we will create a blank/completely black image that has the same shape as the og image
line_img = np.zeros_like(lane_copy2)
#now we will first makes sure that there were actually lines extracted from that image and get the segments of them
if lines is not None:
    for line in lines:
        #we have to reshape every line segment into a 1D array [x1,y1,x2,y2] but it is currently a 4d array
        #reshaping it into a one dimensional array with 4 elements (x1,y1,x2,y2)
        x1, y1, x2, y2 = line.reshape(4)
        #now let us draw each line on that blank image
        #again first argument is the image we are drawing on, 2nd is first endpoint, 3rd is second endpoint, 4th is the color of the line, 5th is line thickness
        cv2.line(line_img, (x1, y1), (x2,y2), (255, 0, 0), 10)
#so now let us add this to the real image
#we do this by adding the two images together using the addeighted method
#it makes sense as to why the background image is black when - when we add it to the real image, the OG pixels of the real image where there are no linens will stay the same because 0 + their intensity is still their intensity

#first parameter is the first image and
#2nd parameter is the first images associated weight - this weight is what each element in that image array with that weight, which will affect the intensity of the pixels
#third parameter is the second image
#fourth parameter is the 2nd images associated weight
#in this case the first images weight is 0.8, decreasing the pixel intensity
#second images weight is 1, so its intensity will be 20% more than the original lane image, so the lines will be much more defined
#fifth is gamma argument - just putting 1, more of a placeholder not making a difference
img_with_lines = cv2.addWeighted(lane_copy2, 0.8, line_img, 1, 1)
cv2.imshow("lines", img_with_lines)
cv2.waitKey(0)
