
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import scipy.ndimage
import math


from Ball import Ball, ball_distance, direction_matches


def findFrameBalls_CNN(frame, frame_balls):
    

    # TODO: using the frame, find ball bounding circles (might have to
    # convert boxes to circles depending on the method). Then, update the
    # frame with the bounding circle before returning it AND initialize
    # ball classes for every bounding circle in the frame before appending
    # that ball to the frame_balls

    return frame