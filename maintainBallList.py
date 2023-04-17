
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import scipy.ndimage
import math


from Ball import Ball, ball_distance, direction_matches


def updateBalls(balls, frame_balls, lines):
    # Cycle through frame_balls to see if balls have changed positions
    leftover_balls_1 = []

    # search all frame_balls for a close match
    for frame_ball in frame_balls:
        match = False

        # if a match exists, the ball didn't move
        for ball in balls:
            if ball_distance(ball, frame_ball) < 1.0:
                ball.update(frame_ball)
                match = True
                break

        if not match: leftover_balls_1.append(frame_ball)


    for frame_ball in leftover_balls_1:
        best_distance = 1e6
        best_ball = 0

        for ball in balls:
            if ball.updated: continue

            distance = ball_distance(ball, frame_ball)

            if distance < best_distance:
                best_distance = distance
                best_ball = ball

        if best_distance < 100 and direction_matches(best_ball, frame_ball):
            # found velocity match
            best_ball.velocity_from_previous(frame_ball)
            best_ball.display(lines)

        else:
            # assume it is a new ball
            balls.append(frame_ball)



    # reset balls for next iteration
    for ball in balls: ball.updated = False