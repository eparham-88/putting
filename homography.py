import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import scipy.ndimage
import math
from common import homography_transform

class BBox:
    def __init__(self,ball):
        self.points = np.array([[ball.positions[0][0] - ball.radii[0], ball.positions[0][1]],
                                [ball.positions[0][0], ball.positions[0][1] + ball.radii[0]],
                                [ball.positions[0][0] + ball.radii[0], ball.positions[0][1]],
                                [ball.positions[0][0], ball.positions[0][1] - ball.radii[0]]])
        self.rad_a = ball.radii[0]
        self.rad_b = ball.radii[0]
        self.area = math.pi * ball.radii[0] * ball.radii[0]

        self.points_old = self.points
        self.rad_a_old = self.rad_a
        self.rad_b_old = self.rad_b
        self.area_old = self.area
    
    def update(self,H):
        self.points = homography_transform(self.points, H)
        self.rad_a = dist(self.points[0], self.points[2])/2
        self.rad_b = dist(self.points[3], self.points[1])/2
        self.area = math.pi * self.rad_a * self.rad_b

    def revert_back(self):
        self.points = self.points_old
        self.rad_a = self.rad_a_old
        self.rad_b = self.rad_b_old
        self.area = self.area_old
    
class BBoxes:
    def __init__(self):
        self.boxes = []
        self.areas = []
        self.mean = 0
        self.stdDev = 0
        self.ratio = 1

    def add(self, balls):
        for ball in balls:
            box = BBox(ball)
            self.boxes.append(box)
            self.areas.append(box.area)
        self.findStats()

    def update(self, H):
        for i in range(len(self.boxes)):
            self.boxes[i].update(H)
            self.areas[i] = self.boxes[i].area
        self.findStats()

    def revert_back(self):
        for i in range(len(self.boxes)):
            self.boxes[i].revert_back()
            self.areas[i] = self.boxes[i].area
        self.findStats()


    def findStats(self):
        self.mean = np.mean(self.areas)
        self.stdDev = np.std(self.areas)
        self.ratio = self.stdDev / self.mean

    def copy(self, other):
        self.boxes = other.boxes
        self.areas = other.areas
        self.mean = other.mean
        self.stdDev = other.stdDev
        self.ratio = other.ratio

   

    


    
def dist(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    return (x**2 + y**2)**(1/2)      

def getHomography_pts_src(frame, iter):
    # TODO: Figure out how to automatically find the four homography input points, or don't
    pts_src = None

    # pts_src = np.array([[156, 162], [108, 245], [695, 235], [583, 155]])
    pts_src = np.array([[156, 152],
                        [108, 245],
                        [695, 235],
                        [583, 147]])
    pts_src[0,1] -= iter
    pts_src[3,1] -= iter

    return pts_src

def getHomography_pts_dst(frame):
    # TODO: Figure out how to automatically choose 4 homography output points, or don't
    pts_dst = None

    pts_dst = np.array([[160, 440], [160, 840], [560, 840], [560, 440]])

    return pts_dst

def runHomography(frame, frame_balls, num_iters):
    boxes = BBoxes()
    boxes.add(frame_balls) 
    # for ball in frame_balls:
    #     box = BBox(ball)
    #     boxes.add(box)

    pts_dst = getHomography_pts_dst(frame)

    iter = 0
    best_ratio = None
    best_boxes = BBoxes()
    best_H = None
    while iter < 1:
    # while iter < num_iters:
        pts_src = getHomography_pts_src(frame, iter)
        H, status = cv2.findHomography(pts_src, pts_dst)
        
        boxes.update(H)
        if (best_ratio == None or (boxes.ratio < best_ratio and boxes.mean > 50 and boxes.mean < 800)):
            best_ratio = boxes.ratio
            best_boxes.copy(boxes)
            best_H = np.copy(H)
        boxes.revert_back()
        iter += 1

    
        
    


    print(best_H)
    return best_H