import cv2
import numpy as np
import matplotlib.pyplot as plt

#first let us capture the video
cap = cv2.VideoCapture("road_video.mp4")
while(cap.isOpened()):
    #read the current frame of the video
    #repeat the process previously detailed for an image, for each frame of the video
    #we only want the frame for now
    _, frame = cap.read()
    lane_copy = np.copy(frame)
    lane_copy2 = np.copy(frame)
    lane_gray = cv2.cvtColor(lane_copy, cv2.COLOR_RGB2GRAY)
    lane_blur = cv2.GaussianBlur(lane_gray, (5, 5), 0)
    canny = cv2.Canny(lane_blur, 50, 150)
    triangle = np.array([[(200, 700), (1100, 700), (500, 250)]])
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, triangle, 255)
    roi = cv2.bitwise_and(canny, mask)
    lines = cv2.HoughLinesP(roi, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    if len(left_fit) > 0:
        slope = left_fit_average[0]
        y_intercept = left_fit_average[1]
        y1 = int(lane_gray.shape[0])
        y2 = int(lane_gray.shape[0] * (3 / 5))
        x1 = int((y1 - y_intercept) / slope)
        x2 = int((y2 - y_intercept) / slope)
        left_line = np.array([x1, y1, x2, y2])
    if len(right_fit) > 0:
        slope = right_fit_average[0]
        y_intercept = right_fit_average[1]
        y1 = int(lane_gray.shape[0])
        y2 = int(lane_gray.shape[0] * (3 / 5))
        x1 = int((y1 - y_intercept) / slope)
        x2 = int((y2 - y_intercept) / slope)
        right_line = np.array([x1, y1, x2, y2])
    averaged_lines = np.array([left_line, right_line])

    line_img = np.zeros_like(lane_copy2)
    if averaged_lines is not None:
        for line in averaged_lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    img_with_lines = cv2.addWeighted(lane_copy2, 0.8, line_img, 1, 1)
    #now show the current frame with the lines
    cv2.imshow("frame_lines", img_with_lines)
    #when we press a certain key we want the video to actually close out
    #in this case that is q
    #wait 1 millisecod between each frame
    #we will break out of the loop when press the letter q
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break
#close the video and destroy the windows when the loop breaks
cap.release()
cv2.destroyAllWindows()


