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

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class VideoProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		faces = cascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)

		for x,y,w,h in faces:
			cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 3)

		return av.VideoFrame.from_ndarray(frm, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)
