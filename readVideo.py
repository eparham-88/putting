import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import scipy.ndimage
import math
# Put all your random shit in common and import it (I've been copying things from 442 into it)
from common import (find_maxima, read_img, visualize_maxima,
                    visualize_scale_space, gaussian_filter,
                    homography_transform)

from Ball import Ball, ball_distance, direction_matches
from findFrameBalls_MaskingMethod import findFrameBalls_masks
from findFrameBalls_CNNMethod import findFrameBalls_CNN
from maintainBallList import updateBalls
from homography import getHomography_pts_src, getHomography_pts_dst, runHomography

balls = []


def readVideo_findHomograph(fname_in, max_frames):
    # Setup cv2 stuff.
    cap = cv2.VideoCapture(fname_in)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # ; print("frame_count = ", frame_count)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # ; print("frame_width = ", frame_width)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # ; print("frame_height = ", frame_height)
    size = (frame_width, frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS) # ; print("fps = ", fps)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter(fname_out, fourcc, fps, size)
    # lines = np.zeros((frame_height, frame_width, 3))

    frame_index = 0
    best_ball_count = None
    best_frame = []
    best_frame_balls = []
    H_matrix = None
    # while loop looks at each frame.
    while (cap.isOpened() and frame_index < max_frames):
        ret, frame = cap.read()
        if not ret:
            # print("Can't receive frame. Exiting...")
            break

        frame_balls = []
        frame = findFrameBalls_masks(frame, frame_balls, frame_index == 0)
        if (best_ball_count == None or len(frame_balls) > best_ball_count):
            best_frame_balls = np.copy(frame_balls)
            best_ball_count = len(frame_balls)
            best_frame = np.copy(frame)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_index += 1
    
    cap.release()
    cv2.destroyAllWindows()

    H = runHomography(best_frame, best_frame_balls, 5000)
    first_frame_out = cv2.warpPerspective(frame, H, size)
    fname_out = 'output/frame_with_most_balls.png'
    cv2.imwrite(fname_out, first_frame_out)

    return H












def readWriteVideo(fname_in, fname_out, H):
    # Setup cv2 stuff.
    cap = cv2.VideoCapture(fname_in)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); print("frame_count = ", frame_count)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); print("frame_width = ", frame_width)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); print("frame_height = ", frame_height)
    size = (frame_width, frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS); print("fps = ", fps)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(fname_out, fourcc, fps, size)
    lines = np.zeros((frame_height, frame_width, 3))

    frame_index = 0
    # while loop looks at each frame.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # print("Can't receive frame. Exiting...")
            break

        frame_balls = []

        if 1:
            # Use the HSV masking approach
            frame_contours = findFrameBalls_masks(frame, frame_balls, frame_index == 0) # find balls, then homography
            # frame_contours = findFrameBalls_masks(cv2.warpPerspective(frame, H, size), frame_balls, frame_index == 0) # homography, then find balls (not tuned correctly yet)

        else:
            # use the CNN approach
            frame_contours = findFrameBalls_CNN(frame, frame_balls)
        

        updateBalls(balls, frame_balls, lines)


        # apply lines
        lines_mask = lines[:,:,0] > 0
        frame_contours[lines_mask] = 0.5*frame_contours[lines_mask] + 0.5*np.array([255, 0, 0]).astype(np.uint8)

        out.write(cv2.warpPerspective(frame_contours, H, size)) # find balls, then homography
        # out.write(frame_contours) # homography, then find balls (not tuned correctly yet)
        if cv2.waitKey(1) == ord('q'):
            break

        # Prints progress
        frame_index += 1; print("frame", frame_index, "/", frame_count, "complete")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # if len(sys.argv) != 3 or ((sys.argv[1])[-4:] != ".MOV" and (sys.argv[1])[-4:] != ".mov" and (sys.argv[1])[-4:] != ".mp4") or (sys.argv[2])[-4:] != ".mp4":
    #     print("\nYour ran this wrong, you friggin fart smeller. You need to run: python readVideo.py inputFilename.MOV outputFilename.mp4\n")
    #     sys.exit(1)

    fname_in = "input videos/IMG_8198.MOV"
    fname_out = "output/8198_output.mp4"
    print("\nInput filename = ", fname_in, "\nOutput filename = ", fname_out, "\n")

    H = readVideo_findHomograph(fname_in, 100)
    readWriteVideo(fname_in, fname_out, H)

    

        