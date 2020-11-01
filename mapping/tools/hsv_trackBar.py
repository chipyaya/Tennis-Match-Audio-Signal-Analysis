import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("Hsv Capture")

# create trackbars for color change
# IMPORTANT: You have to define the correct HSV opencv range hence 179,255,255
cv2.createTrackbar('H', 'Hsv Capture', 0, 179, nothing)
cv2.createTrackbar('S', 'Hsv Capture', 0, 255, nothing)
cv2.createTrackbar('V', 'Hsv Capture', 0, 255, nothing)

cv2.createTrackbar('H1', 'Hsv Capture', 0, 179, nothing)
cv2.createTrackbar('S1', 'Hsv Capture', 0, 255, nothing)
cv2.createTrackbar('V1', 'Hsv Capture', 0, 255, nothing)

img_path = '../video2image/us19-images/00:00-00:07-fps6/001.png'
frame = cv2.imread(img_path)
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
while(True):
    # Trackbars realtime position
    h1 = cv2.getTrackbarPos('H', 'Hsv Capture')
    s1 = cv2.getTrackbarPos('S', 'Hsv Capture')
    v1 = cv2.getTrackbarPos('V', 'Hsv Capture')

    h2 = cv2.getTrackbarPos('H1', 'Hsv Capture')
    s2 = cv2.getTrackbarPos('S1', 'Hsv Capture')
    v2 = cv2.getTrackbarPos('V1', 'Hsv Capture')

    #How to store the min and max values from the trackbars
    blue_MIN = np.array([h1, s1, v1], np.uint8)
    blue_MAX = np.array([h2, s2, v2], np.uint8)

    #After finding your values, you can replace them like this
    #blue_MIN = np.array([102, 73, 145], np.uint8)
    #blue_MAX = np.array([123, 182, 175], np.uint8)
            
    #Using inRange to find the desired range
    hsvCapture = cv2.inRange(hsv_frame, blue_MIN, blue_MAX)

    cv2.imshow('Hsv Capture', hsvCapture)
    # cv2.imshow('Hsv Capture', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
