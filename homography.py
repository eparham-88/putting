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
    ### BBox takes ball and allows for transformation to ellipse, also finds the ellipse's new area ###
    def __init__(self,ball):
        self.points = np.array([[ball.positions[0][0] - ball.radii[0], ball.positions[0][1]],
                                [ball.positions[0][0], ball.positions[0][1] + ball.radii[0]],
                                [ball.positions[0][0] + ball.radii[0], ball.positions[0][1]],
                                [ball.positions[0][0], ball.positions[0][1] - ball.radii[0]]])
        self.rad_a = ball.radii[0]
        self.rad_b = ball.radii[0]
        self.area = math.pi * ball.radii[0] * ball.radii[0]
        # Ratio evaluates how circular the ellipse is (smaller => worse, 1 => perfect circle)
        self.ratio = 1.0

        self.points_old = self.points
        self.rad_a_old = self.rad_a
        self.rad_b_old = self.rad_b
        self.area_old = self.area
        self.ratio_old = self.ratio
    
    def update(self,H):
        self.points = homography_transform(self.points, H)
        self.rad_a = dist(self.points[0], self.points[2])/2
        self.rad_b = dist(self.points[3], self.points[1])/2
        self.area = math.pi * self.rad_a * self.rad_b
        self.ratio = min(float(self.rad_a), float(self.rad_b)) / max(float(self.rad_a), float(self.rad_b), 1)

    def revert_back(self):
        self.points = self.points_old
        self.rad_a = self.rad_a_old
        self.rad_b = self.rad_b_old
        self.area = self.area_old
        self.ratio = self.ratio_old
    
class BBoxes:
    ### BBoxes is a list of Boxes that tracks Box area mean, stdDev, circle_ratio_mean ###
    ### (evaluates how) close ellipses are to circular, and overall score              ###
    def __init__(self):
        self.boxes = []
        self.areas = []
        self.circle_ratios = []
        self.mean = 0
        self.stdDev = 0
        self.circle_ratio_mean = 0
        self.score = 0

    def add(self, balls):
        for ball in balls:
            box = BBox(ball)
            self.boxes.append(box)
            self.areas.append(box.area)
            self.circle_ratios.append(box.ratio)
        self.findStats()

    def update(self, H):
        for i in range(len(self.boxes)):
            self.boxes[i].update(H)
            self.areas[i] = self.boxes[i].area
            self.circle_ratios[i] = self.boxes[i].ratio
        self.findStats()

    def revert_back(self):
        for i in range(len(self.boxes)):
            self.boxes[i].revert_back()
            self.areas[i] = self.boxes[i].area
            self.circle_ratios[i] = self.boxes[i].ratio
        self.findStats()


    def findStats(self):
        self.mean = np.mean(self.areas)
        self.stdDev = np.std(self.areas)
        self.circle_ratio_mean = np.mean(self.circle_ratios)

        self.score = self.mean / self.stdDev + 0.01 * self.circle_ratio_mean # Change 0.01 if you want to prioritize circular shape more

    def copy(self, other):
        self.boxes = other.boxes
        self.areas = other.areas
        self.circle_ratios = other.circle_ratios
        self.mean = other.mean
        self.stdDev = other.stdDev
        self.score = other.score
        self.circle_ratio_mean = other.circle_ratio_mean

   

    


    
def dist(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    return (x**2 + y**2)**(1/2)      

def getHomography_pts_src(frame, iter):
    # TODO: Figure out how to automatically find the four homography input points, or don't
    pts_src = None

    ### OLD APPROACH (works about as good, repeatable) ###
    # pts_src = np.array([[156, 162], [108, 245], [695, 235], [583, 155]])
    # pts_src = np.array([[156, 152],
    #                     [108, 245],
    #                     [695, 235],
    #                     [583, 147]])
    # pts_src[0,1] -= iter
    # pts_src[3,1] -= iter

    ### NEW APPROACH (works better, but not consistent) ###
    scale_x = 5.0; scale_y = 100.0
    average_slope_thresh = 5
    average_slope = None
    while average_slope == None or abs(average_slope) > average_slope_thresh:
        pts_src = np.array([[int(np.random.normal(156, scale_x)), int(np.random.normal(152, scale_y))],
                            [int(np.random.normal(108, scale_x)), int(np.random.normal(245, scale_y))],
                            [int(np.random.normal(695, scale_x)), int(np.random.normal(235, scale_y))],
                            [int(np.random.normal(583, scale_x)), int(np.random.normal(147, scale_y))]])
        average_slope = (math.degrees(math.atan2(pts_src[0,1] - pts_src[3,1], pts_src[0,0] - pts_src[3,0])) +
                         math.degrees(math.atan2(pts_src[1,1] - pts_src[2,1], pts_src[1,0] - pts_src[2,0])))/2
        if (average_slope < 0):
            average_slope += 360
    
    return pts_src

def getHomography_pts_dst(frame):
    # TODO: Figure out how to automatically choose 4 homography output points, or don't
    pts_dst = None

    pts_dst = np.array([[160, 440], [160, 840], [560, 840], [560, 440]])

    return pts_dst

def runHomography(frame, frame_balls, num_iters):
    # Initialize and make a list of boxes
    boxes = BBoxes()
    boxes.add(frame_balls) 

    # These points are hard-coded
    pts_dst = getHomography_pts_dst(frame)

    iter = 0
    best_score = None
    best_boxes = BBoxes()
    best_H = None
    best_iter = 0
    # while iter < 1: # Uncomment to run a single iteration
    while iter < num_iters:
        # Generates new set of source points, finds corresponding H
        pts_src = getHomography_pts_src(frame, iter)
        H, status = cv2.findHomography(pts_src, pts_dst)
        
        # Update the boxes with H and see if the score is better, but also make sure ball
        # mean areas are good.
        boxes.update(H)
        if (best_score == None or (boxes.score > best_score and boxes.mean > 50 and boxes.mean < 1000)):
            # Save stuff if a better H was found
            best_score = boxes.score
            best_boxes.copy(boxes)
            best_H = np.copy(H)
            best_iter = iter

        # Revert the boxes back before running again (instead of constantly making new ones)
        boxes.revert_back()
        iter += 1

    # print("best iteration =", best_iter)
    # print(best_H)
    return best_H