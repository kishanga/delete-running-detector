from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import mediapipe as mp

import tempfile
import streamlit as st
import numpy as np
import csv
import os
import math as m
from PIL import ImageFont, ImageDraw, Image

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pickle
import sounddevice as sd
import pandas as pd
import warnings
import time

import sklearn.metrics
warnings.filterwarnings("ignore")


# mediapipe inbuilt solutions 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model_path = 'finalized_model.sav'

class VideoProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")
		mp_drawing = mp.solutions.drawing_utils
		mp_holistic = mp.solutions.holistic
		mp_pose = mp.solutions.pose
		pose = mp_pose.Pose()
		class_name='Good'
	
		drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

		count=0
	

	    	#model_full_path = os.path.join(os.getcwd(), model_path)
	    	#print(model_full_path)
	
		with open(model_path, 'rb') as f:
	        	clf = pickle.load(f)
	        	count = 0
	  
		with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
	

			#recoloring it back to BGR b/c it will rerender back to opencv
			image = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False
	
			results = holistic.process(image)
			# Process the image.
			# Get height and width.
			h, w = image.shape[:2]
			font_size=int(w/25)-8
			if w<400 and h>400:
			    font_size=int(w/15)-5
	
			# Use lm and lmPose as representative of the following methods.
			lm = results.pose_landmarks
			lmPose = mp_pose.PoseLandmark
	
			mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
							mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
							mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
	
			# Acquire the landmark coordinates.
			# Once aligned properly, left or right should not be a concern.
			try:
			    # Left shoulder.
			    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
			    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
	
			    # Right shoulder
			    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
			    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
	
			    # Left ear.
			    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
			    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
			    # Left hip.
	
			    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
			    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
	
			    # Calculate angle.
			    def findAngle(x1, y1, x2, y2):
				    theta = m.acos((y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
				    degree = int(180/m.pi)*theta
				    return degree
			
			    
			    # setting image writeable back to true to be able process it
			    image.flags.writeable = True
			    pil_im = Image.fromarray(image)
			    draw = ImageDraw.Draw(pil_im)
	
			    # def angle
			    torso_angle = findAngle(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)
			    torso_angle = "Shoulder-Hip Angle:" + str(round(torso_angle,2))
			    neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
			    neck_inclination = "Neck Inclination (Ear-Shoulder Angle):" + str(round(neck_inclination,2))
			    
			    # font
			    font = ImageFont.truetype(font = "Arial-Unicode.ttf", size = font_size)
			    # font = ImageFont.truetype(font = "Arial-Unicode.ttf", size = font_size)
			    
			    # draw text
			    draw.text((20,0), torso_angle, font = font, fill = 'black')
			   
			    temp=0
			    image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
			    temp=temp+font_size+7
			    draw.text((20,temp), neck_inclination, font = font, fill = 'black')
			    temp=temp+temp
			    image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
	
			    # Draw the pose landmarks
			    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
							mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
							mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
			    
			    len(results.pose_landmarks.landmark)
			    num_coords = len(results.pose_landmarks.landmark)
			    landmarks = ['class']
			    for val in range(1, num_coords+1):
				    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
			    with open('Prediction_info.csv', mode='w', newline='') as f:
				    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				    csv_writer.writerow(landmarks)
			    class_name = 'abc'
			except:
			    pass
			try:
			    
			    pose = results.pose_landmarks.landmark
			    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
	
			    row = pose_row 
			    row.insert(0, class_name)
			    
			    with open('Prediction_info.csv', mode='a', newline='') as f:
				    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				    csv_writer.writerow(row)
	
			    df = pd.read_csv('Prediction_info.csv')
			    nose = [df.at[0,'x1'],df.at[0,'y1']]
			    
			    left_eye = [df.at[0,'x3'],df.at[0,'y3']]
			    left_ear = [df.at[0, 'x8'], df.at[0, 'y8']]
			    left_shoulder = [df.at[0, 'x12'], df.at[0, 'y12']]
			    left_hip = [df.at[0, 'x24'], df.at[0, 'y24']]
			    left_knee = [df.at[0, 'x26'], df.at[0, 'y26']]
	
			    right_eye = [df.at[0,'x6'],df.at[0,'y6']]
			    right_ear = [df.at[0, 'x9'], df.at[0, 'y9']]
			    right_shoulder = [df.at[0, 'x13'], df.at[0, 'y13']]
			    right_hip = [df.at[0, 'x25'], df.at[0, 'y25']]
			    right_knee = [df.at[0, 'x27'], df.at[0, 'y27']]
			    
			    left_EyeSH = calculate_angle(left_eye,left_shoulder,left_hip)
			    right_EyeSH = calculate_angle(right_eye, right_shoulder, right_hip)
			    left_ESH = calculate_angle(left_ear,left_shoulder,left_hip)
			    right_ESH = calculate_angle(right_ear, right_shoulder, right_hip) 
			    left_NSH = calculate_angle(nose,left_shoulder,left_hip)
			    right_NSH = calculate_angle(nose, right_shoulder, right_hip)
			    left_SHK = calculate_angle(left_shoulder,left_hip, left_knee)
			    right_SHK = calculate_angle(right_shoulder, right_hip, right_knee)
			    
			    test = [left_EyeSH,right_EyeSH,left_ESH,right_ESH,left_NSH,right_NSH,left_SHK,right_SHK]
			    answer = clf.predict([test])
			    print(answer)
			    
			    print_ans="0"
			    
			    if answer==0:
				    print_ans="Bad Posture"
				    count = count+1
				    
				    # Beep sound
				    if count >= 1 and count <= 5:
					    duration = 0.5  # seconds
					    frequency = 1000  # Hz
					    t = np.linspace(0, duration, int(44100 * duration), endpoint=False)
					    beep = 0.5 * np.sin(2 * np.pi * frequency * t)
					    
					    # Play the beep sound
					    sd.play(beep, samplerate=44100)
					    sd.wait()
			    else:
				    print_ans = "Good Posture"
	
			    
			    if int(font_size/10)<7:
				    ss=font_size+27
			    else:
				    ss=int(font_size/10)+42+font_size
			 
			    font = ImageFont.truetype("Arial-Unicode.ttf", ss)
			    # font = ImageFont.truetype("Arial Unicode.ttf", ss)
			    bbox = draw.textbbox((20,temp), print_ans, font = font)
			    draw.rectangle(bbox, fill = 'black')
			    draw.text((20,temp), print_ans, font = font, fill = 'white')
			    
			    image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
			except:
				pass
	
			#stframe.image(image,use_column_width=True)
	
	
	
	
		return av.VideoFrame.from_ndarray(image, format='bgr24')





	



	
	def calculate_angle(a,b,c):
	    
	    a = np.array(a) # First
	    b = np.array(b) # Mid
	    c = np.array(c) # End
	    # print(c[1])
	    # Calculate angles for all rows using vector operations
	    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
	    angles = np.abs(radians * 180.0 / np.pi)
	        
	    # Ensure angles are within [0, 360) degrees
	    # angles = np.where(angles > 180.0, 360 - angles, angles)
	    if angles>180.0:
	        angles = 360-angles
	        
	    return angles





webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)
