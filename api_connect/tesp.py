import sampleClient
import realsense_sensor as sensor
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
try:
	os.chdir(os.path.join(os.getcwd(), '..'))
	print(os.getcwd())
except:
	pass

cam = sensor.RealsenseSensor("cfg/sensors/realsense_config.json")
cam.start()
