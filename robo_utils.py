#cd ~/examples/lite/examples/object_detection/raspberry_pi
#python ./detect.py --model lite-model_efficientdet_lite2_detection_metadata_1.tflite 
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import vision
from tflite_support.task import processor

import numpy as np
import math
import scipy.optimize  
 
class Arm3Link:
     
    def __init__(self, q=None, q0=None, L=None):
        """Set up the basic parameters of the arm.
        All lists are in order [shoulder, elbow, wrist].
         
        :param list q: the initial joint angles of the arm
        :param list q0: the default (resting state) joint configuration
        :param list L: the arm segment lengths
        """
        # initial joint angles
        if q is None: q = [.3, .3, 0]
        self.q = q
        # some default arm positions
        if q0 is None: q0 = np.array([math.pi/4, math.pi/4, math.pi/4]) 
        self.q0 = q0
        # arm segment lengths
        if L is None: L = np.array([1, 1, 1]) 
        self.L = L
         
        self.max_angles = [math.pi, math.pi, math.pi/4]
        self.min_angles = [0, 0, -math.pi/4]
 
    def get_xy(self, q=None):
        if q is None: q = self.q
 
        x = self.L[0]*np.cos(q[0]) + \
            self.L[1]*np.cos(q[0]+q[1]) + \
            self.L[2]*np.cos(np.sum(q))
 
        y = self.L[0]*np.sin(q[0]) + \
            self.L[1]*np.sin(q[0]+q[1]) + \
            self.L[2]*np.sin(np.sum(q))
 
        return [x, y]
 
    def inv_kin(self, xy):
 
        def distance_to_default(q, *args): 
            # weights found with trial and error, get some wrist bend, but not much
            weight = [1, 1, 1.3] 
            return np.sqrt(np.sum([(qi - q0i)**2 * wi
                for qi,q0i,wi in zip(q, self.q0, weight)]))
 
        def x_constraint(q, xy):
            x = ( self.L[0]*np.cos(q[0]) + self.L[1]*np.cos(q[0]+q[1]) +
                self.L[2]*np.cos(np.sum(q)) ) - xy[0]
            return x
 
        def y_constraint(q, xy): 
            y = ( self.L[0]*np.sin(q[0]) + self.L[1]*np.sin(q[0]+q[1]) +
                self.L[2]*np.sin(np.sum(q)) ) - xy[1]
            return y
         
        return scipy.optimize.fmin_slsqp( func=distance_to_default, 
            x0=self.q, eqcons=[x_constraint, y_constraint], 
            args=[xy], iprint=0) # iprint=0 suppresses output
    



_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red


def filter_and_visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
    min_score: float,
    obj: str
) -> (np.ndarray, list):
  results = []
  for detection in detection_result.detections:
    category = detection.categories[0]
    category_name = category.category_name
    if obj == category_name and category.score > min_score:
        results.append(detection)
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + bbox.origin_x,
                         _MARGIN + _ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

  return image, results


def preprocess_image(image):
    matrix = np.array([[643.404,0,285.5126],[0,643.0329,228.4627],[0,0,1]])
    distortion = np.array([[-0.34145, -0.0315, 0.0021, -0.0055, 0.5474]])
    
    
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h))
    image = cv2.undistort(image, matrix, distortion, None, newcameramtx)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    
    return input_tensor
