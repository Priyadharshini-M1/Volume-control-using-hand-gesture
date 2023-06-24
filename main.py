import cv2
import numpy as np
import math
import pyautogui
import time

# Define lower and upper bounds of skin color in HSV format
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Set up camera capture
cap = cv2.VideoCapture(0)

# Initialize previous volume to 0
prev_vol = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to isolate the skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a blur to the mask to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)

        # Calculate the center of the contour
        moments = cv2.moments(max_contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            # Draw a circle at the center of the contour
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # Find the distance from the center of the contour to the bottom of the frame
            distance_to_bottom = frame.shape[0] - cy

            # Map the distance to a volume level (from 0 to 100)
            volume = int(np.interp(distance_to_bottom, [0, frame.shape[0]], [0, 100]))

            # Set the volume using pyautogui (if it has changed)
            if volume != prev_vol:
                pyautogui.press('volumedown', presses=prev_vol - volume)
                pyautogui.press('volumeup', presses=volume - prev_vol)
                prev_vol = volume

    # Show the frame with the contour and volume level
    cv2.imshow('frame', frame)

    # Check for keypresses (to exit the loop)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()