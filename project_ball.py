import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from Ball import Ball
import matplotlib.pyplot as plt
from readVideo import readWriteVideo, balls
from vector_field import Path





if __name__ == "__main__":

    # load balls
    with open('vector_field.pkl', 'rb') as f:
        [vector_field, friction_acceleration] = pickle.load(f)

    with open('homography.pkl', 'rb') as f:
        H = pickle.load(f)

    
    fname_in = "input videos/In.MOV"
    fname_out = "output/In.mp4"


    readWriteVideo(fname_in, fname_out, H, final=True)

    path = Path(balls[3])

    x_pos = path.waypoints[1][2]
    y_pos = path.waypoints[1][3]

    x_vel = path.velocities[1][0]; x_prev_vel = path.velocities[0][0]
    y_vel = path.velocities[1][1]; y_prev_vel = path.velocities[0][1]

    cell_size = int(720 / vector_field.shape[0])

    # update video with prediction
    cap = cv2.VideoCapture(fname_in)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); print("frame_count = ", frame_count)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); print("frame_width = ", frame_width)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); print("frame_height = ", frame_height)
    size = (frame_width, frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS); print("fps = ", fps)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter("output/prediction.mp4", fourcc, fps, size)
    lines = np.zeros((frame_height, frame_width, 3))
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # print("Can't receive frame. Exiting...")
            break

        frame = cv2.warpPerspective(frame, H, size)

        if frame_index > 27 and y_pos > 600:


            x_index = min(round(x_pos/cell_size), int(frame.shape[0]/cell_size)-1)
            y_index = min(round(y_pos/cell_size), int(frame.shape[1]/cell_size)-1)


            den = math.sqrt(x_prev_vel**2 + y_prev_vel**2)
            if den == 0: den = 1

            # figure out previous velocity vector direction to apply friction to
            x_prev_vel = x_prev_vel / den
            y_prev_vel = y_prev_vel / den

            # apply friction
            x_vel -= x_prev_vel*friction_acceleration # 2
            y_vel -= y_prev_vel*friction_acceleration

            # apply vector field
            x_vel += 5*vector_field[x_index, y_index][0] # 0.4
            y_vel += vector_field[x_index, y_index][1]

            cv2.line(lines, (int(x_pos), int(y_pos)), (int(x_pos+x_vel), int(y_pos+y_vel)), (0,0,255), 5)


            # update position
            x_pos += x_vel
            y_pos += y_vel

            # prepare for next iteration
            x_prev_vel = x_vel
            y_prev_vel = y_prev_vel



        lines_mask = lines[:,:,2] > 0
        frame[lines_mask] = 0.5*frame[lines_mask] + 0.5*np.array([0, 0, 255]).astype(np.uint8)

        out.write(frame)

        frame_index += 1




    cap.release()
    out.release()
    cv2.destroyAllWindows()







