
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import scipy.ndimage
import math
import torch
from roboflow import Roboflow


from Ball import Ball, ball_distance, direction_matches

def init_CNN_pretrained():
    # API ping for open source golfball dataset and trained model

    rf = Roboflow(api_key="VY0RgvVMmLt6Rzqqwlnj")
    project = rf.workspace("anna-gaming").project("golfball")
    model = project.version(1).model

    return model

def init_CNN_ours():
    # API ping for our golfball dataset and trained model

    rf = Roboflow(api_key="VY0RgvVMmLt6Rzqqwlnj")
    project = rf.workspace("golf-ball-detector-v2").project("golf-ball-tracker-v2")
    model = project.version(2).model

    return model

def findFrameBalls_CNN(frame, frame_balls, index, model):

    frame_contours = frame.copy()

    predictions_raw = model.predict(frame, confidence=2)
    predictions = predictions_raw.predictions

    for prediction in predictions:
        x = prediction.json_prediction['x']
        y = prediction.json_prediction['y']
        w = prediction.json_prediction['width']
        h = prediction.json_prediction['height']
        conf = prediction.json_prediction['confidence']

        radius = np.min([h/2,w/2])

        ball = Ball((x,y), radius, index)
        frame_balls.append(ball)

        cv2.circle(frame_contours,(int(x), int(y)),int(radius),(255,0,0),1)
        cv2.putText(frame_contours, str("radius="+"{:.1f}".format(radius)),
                    (int((x)-45), int((y)+radius+20)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255,0,0))
        cv2.putText(frame_contours, str("pos=("+"{:.1f}".format(x)+","+"{:.1f}".format(y)+")"),
                    (int((x)-45), int((y)+radius+35)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255,0,0))
        cv2.putText(frame_contours, str("confidence="+"{:.1f}".format(conf)),
                    (int((x)-45), int((y)+radius+50)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255,0,0))
        
    return frame_contours