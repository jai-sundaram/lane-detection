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


#now looking out the lines that we drew on the image, we can see that there are multiple different lines for both sides, which kinda looks weird
#instead, we can just have 2 lines (one for each side of the lane) by averaging the y intercept and slope of all the lines
left_fit = []
right_fit = []
#left_fit will contain the averaged lines on the left side of the lane
#right_fit will contain the averages lines on the right side of the lane
for line in lines:
    #we will again do the same thing where we make each line segment 1d (x1,y1,x2,y2) like we did before
    #reshaping it
    x1,y1,x2,y2 = line.reshape(4)
    # we want to pretty much get parameters like the y intercept and slope
    #using the polyfit function it will fit in a polynomial function with form y = mx+b to the x and y points
    #then it will return a vector of coefficients that describe the y intercept
    #the first parameter are the set of x coordinates
    #second parameter are the set of y coordinates
    #third parameter is the degree of the polynomial - in this case it is of first degree
    parameters = np.polyfit((x1,x2), (y1,y2),1)
    #first index in return value is slope, y intercept is 2nd index
    slope = parameters[0]
    intercept = parameters[1]
    #now we have to check if the slope of the line corresponds to a line on the left side or right side
    #remember that in this case, the y values decrease as we go towards the bottom
    #on the left side of the lane, as x increases the line goes up (meaning y decreases)
    #on the right side of the line, as x increase the line goes down
    #so lines on the left side of the lane will have a negative slope, lines on the right side of the lane will have a positive slope
    #so now, we will start appending each lines slope and y intercept as a tuple to either the left or right side lines, based on value of slope
    if slope < 0:
        left_fit.append((slope, intercept))
    else:
        right_fit.append((slope, intercept))
#now let us average the slopes and intercepts for both sides
#axis = 0 will average all the rows in each column
left_fit_average  = np.average(left_fit, axis = 0)
right_fit_average = np.average(right_fit, axis = 0)
#this will contain the average slope and y intercept of the overall line we will eventually draw for left and right side
#now we have to create the (x1,y1) and (x2,y2) coordinates for the two final lines from the slope and y intercept
#doing for the left side first
slope = left_fit_average[0]
y_intercept = left_fit_average[1]
#y1 will be the height of the image (bottom of the image)
#get the height by accesing the first index from the shape function of the image
y1 = int(lane_gray.shape[0])
#y2 will be about 60% of the images height
y2 = int(lane_gray.shape[0] * (3/5))
#so now the line starts from the bottom and goes 3/ths of the way upwards
#now we want to get the corresponding x1, and x2 values for the y1, y2 values
#we can figure out this algebraically
#y=mx+b
# x = (y-b)/m
#do this for both x1 and x2
x1 = int((y1-y_intercept) / slope)
x2 = int((y2-y_intercept) / slope)
left_line = np.array([x1,y1, x2, y2])
#repeat the same process for the right line
slope = right_fit_average[0]
y_intercept = right_fit_average[1]
y1 = int(lane_gray.shape[0])
y2 = int(lane_gray.shape[0] * (3/5))
x1 = int((y1-y_intercept) / slope)
x2 = int((y2-y_intercept) / slope)
right_line = np.array([x1,y1, x2, y2])
#we can see that both lines will have the same vertical coordinates - they will start at the very bottom
#they will both go upwards 3/5ths of the image up
#but their horizontal coordinates will be different because they are dependent on the slope and y intercept
#now we have to put these x1 y1 x2 y2 of both lines into one combined numpy array so it matches the HoughLinesP function output
averaged_lines = np.array([left_line, right_line])

#now we want to display these lines on the real/original image
#first we will create a blank/completely black image that has the same shape as the og image
line_img = np.zeros_like(lane_copy2)
#now we will first makes sure that there were actually lines extracted from that image and get the segments of them
if averaged_lines is not None:
    for line in averaged_lines:
        #we need x1 y1 x2 y2 for the endpoints
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



