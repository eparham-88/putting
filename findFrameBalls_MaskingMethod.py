
import numpy as np
import cv2


from Ball import Ball, ball_distance, direction_matches



def filter_radius(y, radius):

    # 3.8 at 176
    # 13.1 at 786.5
    expected_radius = 0.0152334*y + 1.1189

    return abs(expected_radius - radius) / expected_radius > 0.3 or radius < 3.0



def findFrameBalls_masks(frame, frame_balls, print_intermediate_frames=False):
    # Do stuff to frame_out to change the output video
    # frame_out= cv2.warpPerspective(frame, H, size)
    frame_LAB = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower = [0,    0,  80] # [  0,   0,  80]
    upper = [360,  50, 100] # [360,  30, 100]

    lower_np = np.array([(lower[0]/360)*255,
                            (lower[1]/100)*255,
                            (lower[2]/100)*255]).astype(int)
    
    upper_np = np.array([(upper[0]/360)*255,
                            (upper[1]/100)*255,
                            (upper[2]/100)*255]).astype(int)


    # range, mask, blur, threshold
    range = cv2.inRange(frame_LAB, lower_np, upper_np)
    mask = range > 0
    gray = np.zeros_like(mask, np.uint8); gray[mask] = 255
    blurred = cv2.blur(gray, (3,3))
    _, threshold = cv2.threshold(blurred, 150, 255, cv2.THRESH_OTSU)

    frame_mask = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    frame_blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    frame_threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    frame_contours = frame.copy() # frame_contours = frame_threshold.copy()


    # contour
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        
        # find enclosing circle and density
        (x,y),radius = cv2.minEnclosingCircle(contour)
        pixel_area = cv2.contourArea(contour)
        density = 100 * float(pixel_area) / float(np.pi*radius**2)

        if threshold[int(y),int(x)] == 0: continue

        if density < 60: continue

        if filter_radius(y, radius): continue

        ball = Ball((x,y), radius)
        frame_balls.append(ball)

        cv2.circle(frame_contours,(int(x), int(y)),int(radius),(255,0,0),2)
        cv2.putText(frame_contours, str("radius="+"{:.1f}".format(radius)),
                    (int(x-45), int(y+radius+20)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255,0,0))
        cv2.putText(frame_contours, str("pos=("+"{:.1f}".format(x)+","+"{:.1f}".format(y)+")"),
                    (int(x-45), int(y+radius+35)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255,0,0))
        cv2.putText(frame_contours, str("density="+"{:.1f}".format(density)),
                    (int(x-45), int(y+radius+50)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255,0,0))
        

    if print_intermediate_frames:
            cv2.imwrite('output/frame.png', frame)
            cv2.imwrite('output/frame_mask.png', frame_mask)
            cv2.imwrite('output/frame_blurred.png', frame_blurred)
            cv2.imwrite('output/frame_threshold.png', frame_threshold)
            cv2.imwrite('output/frame_contours.png', frame_contours)
            print("shit")

    return frame_contours