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






if __name__ == "__main__":
    # if len(sys.argv) != 3 or ((sys.argv[1])[-4:] != ".MOV" and (sys.argv[1])[-4:] != ".mov" and (sys.argv[1])[-4:] != ".mp4") or (sys.argv[2])[-4:] != ".mp4":
    #     print("\nYour ran this wrong, you friggin fart smeller. You need to run: python readVideo.py inputFilename.MOV outputFilename.mp4\n")
    #     sys.exit(1)

    fname_in = cv2.imread("input frames/test_4.png")

    balls = []

    findFrameBalls_masks(fname_in, balls, True)


    

        