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


def getHomography_pts_src(frame):
    # TODO: Figure out how to automatically find the four homography input points, or don't
    pts_src = None

    pts_src = np.array([[156, 162], [108, 245], [695, 235], [583, 155]])
    # pts_src = np.array([[156, 162], [108, 245], [695, 235], [583, 147]])

    return pts_src

def getHomography_pts_dst(frame, pts_src):
    # TODO: Figure out how to automatically choose 4 homography output points, or don't
    pts_dst = None

    pts_dst = np.array([[160, 440], [160, 840], [560, 840], [560, 440]])

    return pts_dst

def blobDetection(image):
    sigma_1, sigma_2 = 5/(2**(1/2)), 6/(2**(1/2))
    gauss_1 = gaussian_filter(image, sigma_1)
    gauss_2 = gaussian_filter(image, sigma_2)

    # calculate difference of gaussians
    DoG_small = gauss_2 - gauss_1  # to implement

    maxima = find_maxima(DoG_small, k_xy=10)
    image = visualize_maxima(image, maxima, sigma_1, sigma_2/sigma_1)

    return image




balls = []







def readWriteVideo(fname_in, fname_out):
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
    H_matrix = None
    # while loop looks at each frame.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # print("Can't receive frame. Exiting...")
            break

        if frame_index == 0:
            # Do stuff with the very first frame. This is a good place to test ball detection (overwrite frame_test.png)
            # cv2.imwrite('output/first_frame.png', frame)
            # cv2.imwrite('output/frame_test.png', blobDetection(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:,:,0]))
            pts_src = getHomography_pts_src(frame)
            pts_dst = getHomography_pts_dst(frame, pts_src)
            H, status = cv2.findHomography(pts_src, pts_dst)
            first_frame_out = cv2.warpPerspective(frame, H, size)
            cv2.imwrite('output/first_frame_homographied.png', first_frame_out)

        frame_balls = []

        if 1:
            # Use the HSV masking approach
            frame_contours = findFrameBalls_masks(frame, frame_balls, frame_index == 0)

        else:
            # use the CNN approach
            frame_contours = findFrameBalls_CNN(frame, frame_balls)
        

        updateBalls(balls, frame_balls, lines)


        # apply lines
        lines_mask = lines[:,:,0] > 0
        frame_contours[lines_mask] = 0.5*frame_contours[lines_mask] + 0.5*np.array([255, 0, 0]).astype(np.uint8)


        out.write(frame_contours)
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

    readWriteVideo(fname_in, fname_out)

    

        