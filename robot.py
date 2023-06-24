from gpiozero import Robot
from gpiozero import Motor
from gpiozero import Servo
import RPi.GPIO as GPIO
import time
import robo_utils
import utils
from tflite_support.task import core
from tflite_support.task import vision
from tflite_support.task import processor
import cv2
import math



class robot:
    def __init__(self, \
				GPIO_dc_left_rearA, \
				GPIO_dc_left_rearB, \
				GPIO_dc_right_rearA, \
				GPIO_dc_right_rearB, \
                GPIO_sv_claw, \
                GPIO_sv_a0, \
                GPIO_sv_a1, \
                GPIO_sv_a2):
					
        correction = 0.5
        maxPW=(2.0 + correction)/1000
        minPW=(1.0 - correction)/1000            
        self.rear_drive = Robot((GPIO_dc_left_rearA,GPIO_dc_left_rearB),(GPIO_dc_right_rearA,GPIO_dc_right_rearB))
        self.servo_claw = Servo(GPIO_sv_claw)
        self.servo_arm0 = Servo(GPIO_sv_a0, min_pulse_width=minPW, max_pulse_width=maxPW)
        
        correction = 0.55
        maxPW40=(2.0 + correction)/1000
        minPW40=(1.0 - correction)/1000  
        self.servo_arm2 = Servo(GPIO_sv_a2, min_pulse_width=minPW40, max_pulse_width=maxPW40)
        self.servo_arm1 = Servo(GPIO_sv_a1, min_pulse_width=minPW, max_pulse_width=maxPW)

        
        #self.arm = robo_utils.Arm3Link(L=[5,5,5])
        #q = self.arm.inv_kin(xy=(9.64,8))
        #print(q/2/3.1415*360)
        self.current_pos = {"x": 1, "y": 5}
        
        base_options = core.BaseOptions(file_name="lite-model_efficientdet_lite2_detection_metadata_1.tflite", use_coral=False, num_threads=4)
  
        detection_options = processor.DetectionOptions(max_results=15, score_threshold=0.3, category_name_allowlist=["bottle","refrigerator","person"])
  
        options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)

        self.detector = vision.ObjectDetector.create_from_options(options)
        
    def detect(self, obj, score_thresh):
        print("detect")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cnt = 0
        detections = []
        image = None
        while cap.isOpened() and cnt < 10:
            success, image = cap.read()
            if not success:
              sys.exit(
                  'ERROR: Unable to read from webcam. Please verify your webcam settings.'
              )

            cnt += 1
                        
            input_tensor = robo_utils.preprocess_image(image)
            
            base_options = core.BaseOptions(file_name="lite-model_efficientdet_lite2_detection_metadata_1.tflite", use_coral=False, num_threads=4)
  
            detection_options = processor.DetectionOptions(max_results=15, score_threshold=0.2, category_name_allowlist=["bottle","refrigerator","person"])
  
            options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
  
            detector = vision.ObjectDetector.create_from_options(options)
            
            detection_result = detector.detect(input_tensor)
            
            print(detection_result)
            
            #results = self.detector.detect(image)
            image, detections = robo_utils.filter_and_visualize(image, detection_result, score_thresh, obj)
            #image = utils.visualize(image, detection_result)
            cv2.imshow("dect", image)
            if cv2.waitKey(1) == 27:
              break

        cap.release()
        cv2.destroyAllWindows()
        return image, detections

    def drive_forward(self):
        self.rear_drive.forward()
    
    def drive_backward(self):
        self.rear_drive.backward()

    def drive_stop(self):
        self.rear_drive.stop()

    def drive_turn_right(self):
        self.rear_drive.right()
        
    def drive_turn_left(self):
        self.rear_drive.left()
        
    def find_fridge(self):
        fridge_box = None
        while not fridge_box:
            self.drive_turn_right()
            time.sleep(0.5)
            image, fridge_boxes = self.detect("refrigerator", 0.5)
            cv2.imshow("fridge", image)
            if cv2.waitKey(1) == 27:
              break
            time.sleep(5)
            cv2.destroyAllWindows()
            
        print("found fridge")
        
    def close_claw(self):
        self.servo_claw.min()
        time.sleep(1)
            
    def open_claw(self):
        self.servo_claw.max()
        time.sleep(1)
        
    def move_arm0(self, deg):
        self.servo_arm0.value = deg
        time.sleep(2)
            
    def move_arm1(self, deg):
        self.servo_arm1.value = deg
        
    def move_arm2(self, deg):
        self.servo_arm2.value = deg
        
    def straight_arm(self):
        self.servo_arm1.value = 0
        self.servo_arm0.value = 0
        self.servo_arm2.value = 0
        self.close_claw()
        time.sleep(2)
        
    def arm_to_xy(self, x, y):
        #q = self.arm.inv_kin(xy=(x,y))
        #print(q*2/3.1415)
        if x == 0:
            return
        l = 5
        q2 = math.acos((x**2 + y**2 - l*l - l*l)/(2*l*l))
        q1 = math.atan(y/x) - math.atan(l*math.sin(q2)/(2*l*l))
        
        
        
        print("x: " + str(x) + " y: " + str(y))
        print(q1 * 180 / 3.1415)
        print(q2 * 180 /3.1415)

        self.servo_arm1.value = q1 * 4 / 3.1415 / 3
        time.sleep(0.05)

        self.servo_arm2.value = -q2 * 4 / 3.1415 / 3
        time.sleep(0.05)
        
        self.servo_arm0.value = -(q2-q1) * 2 / 3.1415 + 0.1
        time.sleep(0.05)


    def retrieve_beer(self):
        self.find_fridge()
        #self.goto_fridge()
        #self.open_fridge()
        #self.grab_beer()
        #self.close_fridge()
        

#print("starting")  
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(17, GPIO.OUT)
#GPIO.setup(4, GPIO.OUT)
#GPIO.output(17, GPIO.HIGH)
#GPIO.output(4, GPIO.HIGH)
#time.sleep(5)
#GPIO.cleanup()
#print("ending")
beerobo = robot(17, 4, 3, 2, 21, 20, 16, 26)
beerobo.drive_stop()
beerobo.move_arm2(-0.85)
beerobo.arm_to_xy(0, 5)

beerobo.open_claw()

for j in range(100):
    beerobo.arm_to_xy(0+j/15, 5)
    
beerobo.close_claw()

for i in range(100):
    beerobo.arm_to_xy(7-i/15, 5)

time.sleep(55)
beerobo.move_arm2(-0.75)
beerobo.move_arm2(-0.5)
beerobo.move_arm2(-0.25)
beerobo.move_arm2(-0.85)


#beerobo.straight_arm()
beerobo.move_arm0(-0.5)
beerobo.move_arm1(0.5)
beerobo.close_claw()
beerobo.move_arm0(1)
beerobo.move_arm1(0.65)
beerobo.open_claw()
beerobo.move_arm0(0)
beerobo.move_arm1(0)
beerobo.close_claw()
beerobo.move_arm0(-0.5)
beerobo.move_arm1(0.25)



#beerobo.close_claw()
#beerobo.retrieve_beer()

