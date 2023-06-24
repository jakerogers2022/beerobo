import argparse
import sys
import time

import cv2

import glob


# Define the dimensions of checkerboard
CHECKERBOARD = (7, 9)
criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def run():
  
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  i = 0
  
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    i+=1
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # Find the chess board corners
    # If desired number of corners are
    # found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
    				grayColor, CHECKERBOARD, None)   
    print(corners)
    # If desired number of corners can be detected then,
    # refine the pixel coordinates and display
    # them on the images of checker board
    if ret == True:
     cv2.imwrite("./imgs/{}.jpg".format(i), image)    
     time.sleep(3)
     corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
     image = cv2.drawChessboardCorners(image,
										CHECKERBOARD,
										corners2, ret)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  run()


if __name__ == '__main__':
  main()