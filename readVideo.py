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
import pickle

from Ball import Ball, ball_distance, direction_matches
from findFrameBalls_MaskingMethod import findFrameBalls_masks, findFrameBalls_masks_H, findFrameBalls_masks_F
from findFrameBalls_CNNMethod import findFrameBalls_CNN, init_CNN_ours, init_CNN_pretrained
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
        frame = findFrameBalls_masks(frame, frame_balls, frame_index)
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

    frame_edges = np.array([[                  0, 0                  ],
                            [best_frame.shape[1], 0                  ],
                            [0                  , best_frame.shape[0]],
                            [best_frame.shape[1], best_frame.shape[0]]])
    
    frame_edges_h = homography_transform(frame_edges, H)

    size_h = [int(np.max(frame_edges_h[:,1]))+1-int(np.min(frame_edges_h[:,1])),
            int(np.max(frame_edges_h[:,0]))+1-int(np.min(frame_edges_h[:,0]))]
    
    H_translate = np.array([[1, 0, -np.min(frame_edges_h[:,0])],
                            [0, 1, -np.min(frame_edges_h[:,1])],
                            [0, 0, 1]]).astype(float)
    
    H = H_translate @ H

    frame_edges_h = homography_transform(frame_edges, H)

    size_h = [int(np.max(frame_edges_h[:,1]))+1-int(np.min(frame_edges_h[:,1])),
            int(np.max(frame_edges_h[:,0]))+1-int(np.min(frame_edges_h[:,0]))]

    scale = min( best_frame.shape[0] / size_h[0], best_frame.shape[1] / size_h[1] )

    H_scale = np.array([[scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, 1]]).astype(float)

    H = H_scale @ H
    first_frame_out = cv2.warpPerspective(frame, H, size)
    fname_out = 'output/frame_with_most_balls.png'
    cv2.imwrite(fname_out, first_frame_out)

    return H


def readWriteVideo(fname_in, fname_out, H=np.array([]), final=False):
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
    CNN_ours = 1
    if CNN_ours:
        model = init_CNN_ours()
    else:
        model = init_CNN_pretrained()

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
            if not H.any():
                frame_contours = findFrameBalls_masks(frame, frame_balls, frame_index) # find balls, then homography
            elif not final:
                frame_contours = findFrameBalls_masks_H(cv2.warpPerspective(frame, H, size), frame_balls, frame_index)
            else:
                frame_contours = findFrameBalls_masks_F(cv2.warpPerspective(frame, H, size), frame_balls, frame_index)
        else:
            # use the CNN approach
            frame_contours = findFrameBalls_CNN(frame, frame_balls, frame_index, model)
        
        updateBalls(balls, frame_balls, lines, frame_index)


        # apply lines
        lines_mask = lines[:,:,0] > 0
        frame_contours[lines_mask] = 0.5*frame_contours[lines_mask] + 0.5*np.array([255, 0, 0]).astype(np.uint8)

        # out.write(cv2.warpPerspective(frame_contours, H, size)) # find balls, then homography
        out.write(frame_contours) # homography, then find balls (not tuned correctly yet)
        if cv2.waitKey(1) == ord('q'):
            break

        # Prints progress
        frame_index += 1; print("frame", frame_index, "/", frame_count, "complete")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return frame_balls


if __name__ == "__main__":

    names_in = ["IMG_8199",
                "IMG_8196",
                "IMG_8197",
                "IMG_8198",
                "IMG_8200",
                "IMG_8203"]

    
    

    H = readVideo_findHomograph("input videos/IMG_8199.MOV", 100)
    with open('homography.pkl', 'wb') as f:
        pickle.dump(H, f)

    detected_balls = []

    for name in names_in:
        balls = []

        fname_in = "input videos/" + name + ".MOV"
        fname_out = "output/" + name + ".mp4"

        print("\nInput filename = ", fname_in, "\nOutput filename = ", fname_out, "\n")

        clip_balls = readWriteVideo(fname_in, fname_out, H)

        for ball in balls:
            if ball.displayed:
                detected_balls.append(ball)

    with open('balls.pkl', 'wb') as f:
        pickle.dump(detected_balls, f)
    

        