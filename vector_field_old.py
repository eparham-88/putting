import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from Ball import Ball
import matplotlib.pyplot as plt

class Path:

    def __init__(self, ball):
        self.velocities = []
        self.waypoints = []
        self.average_acceleration = 0.0

        if len(ball.positions) < 2: return

        # compute waypoint velocities
        for i in range(1, len(ball.positions)):
            dx = ball.positions[i][0] - ball.positions[i-1][0]
            dy = ball.positions[i][1] - ball.positions[i-1][1]
            dt = float(ball.indices[i] - ball.indices[i-1])

            # normalize for time
            dx = dx / dt
            dy = dy / dt

            # save
            self.velocities.append([
                dx, dy,                                          # velocities
                ball.positions[i-1][0], ball.positions[i-1][1],  # positions
                ball.indices[i-1]                                # time
            ])

        # compute waypoint accelerations
        for i in range(1, len(self.velocities)):
            ddx = self.velocities[i][0] - self.velocities[i-1][0]
            ddy = self.velocities[i][1] - self.velocities[i-1][1]
            ddt = self.velocities[i][4] - self.velocities[i-1][4]

            # normalize for time
            ddx = ddx / ddt
            ddy = ddy / ddt

            # save
            self.waypoints.append([
                ddx,
                ddy,
                self.velocities[i][2],
                self.velocities[i][3]
            ])

        # filter out low accelerations while computing the average
        for waypoint in self.waypoints:
            acceleration_norm = math.sqrt(waypoint[0]**2 + waypoint[1]**2)

            if acceleration_norm < 0.01 or acceleration_norm > 40.0:
                self.waypoints.remove(waypoint)
            else:
                self.average_acceleration += acceleration_norm

        if len(self.waypoints) == 0: return

        self.average_acceleration = self.average_acceleration / float(len(self.waypoints))
            



    def apply_friction(self,friction):
        for waypoint in self.waypoints:

            # use velocity to determine direction of friction
            prev_vel_x = waypoint[0]
            prev_vel_y = waypoint[1]
            den = math.sqrt(prev_vel_x**2 + prev_vel_y**2)
            if den == 0: den = 1

            prev_vel_x = prev_vel_x / den
            prev_vel_y = prev_vel_y / den

            # update acceleration by applying friction in opposite direction of velocity
            waypoint[2] = waypoint[2] - prev_vel_x*friction
            waypoint[3] = waypoint[3] - prev_vel_y*friction






if __name__ == "__main__":

    # load balls
    with open('balls.pkl', 'rb') as f:
        balls = pickle.load(f)

    # Use balls to compute paths and average friction
    average_acceleration = 0.0
    paths = []

    for ball in balls:
        path = Path(ball)
        if path.average_acceleration == 0.0:
            continue
        else:
            average_acceleration += path.average_acceleration
            paths.append(path)
    
    average_acceleration = 0.9 * average_acceleration / float(len(paths))
    print(average_acceleration)

    # adjust for friction, build lists vector field
    x = []; y = []; u = []; v = []
    for path in paths:
        # path.apply_friction(average_acceleration)

        for waypoint in path.waypoints:
            u.append(waypoint[0])
            v.append(waypoint[1])
            x.append(waypoint[2])
            y.append(waypoint[3])




    # plot raw vector field
    plt.quiver(x,y,u,v)
    plt.show()


    # channels are dx, dy, count
    cell_size = 80
    vector_field = np.zeros((int(720/cell_size), int(1280/cell_size), 3))
    x = []; y = []; u = []; v = []

    for path in paths:

        for waypoint in path.waypoints:
            x_index = min(round(waypoint[2]/cell_size), int(720/cell_size)-1)
            y_index = min(round(waypoint[3]/cell_size), int(1280/cell_size)-1)

            vector_field[x_index, y_index] += [waypoint[0], waypoint[1], 1]


    # fix waypoints with more than one point
    more_than_one = vector_field[:,:,2] > 1

    vector_field[more_than_one] = vector_field[more_than_one] / np.vstack((vector_field[more_than_one][:,2], vector_field[more_than_one][:,2], vector_field[more_than_one][:,2])).T

    


    # fill in nearest neighbors
    for i in range(vector_field.shape[0]):
        for j in range(vector_field.shape[1]):

            if vector_field[i, j][2] != 0: continue
            
            # left neighbor
            l_i = i; l_j = j
            while(l_j > 0):
                l_j -= 1
                if vector_field[l_i, l_j][2] != 0:
                    break
            
            # right neighbor
            r_i = i; r_j = j
            while(r_j < vector_field.shape[1]-1):
                r_j += 1
                if vector_field[r_i, r_j][2] != 0:
                    break
            
            # up neighbor
            u_i = i; u_j = j
            while(u_i > 0):
                u_i -= 1
                if vector_field[u_i, u_j][2] != 0:
                    break
            
            # down neighbor
            d_i = i; d_j = j
            while(d_i < vector_field.shape[0]-1):
                d_i += 1
                if vector_field[d_i, d_j][2] != 0:
                    break
            
            dx = vector_field[l_i, l_j][0] + vector_field[r_i, r_j][0] + vector_field[u_i, u_j][0] + vector_field[d_i, d_j][0] / (vector_field[l_i, l_j][2] + vector_field[r_i, r_j][2] + vector_field[u_i, u_j][2] + vector_field[d_i, d_j][2])

            dy = vector_field[l_i, l_j][1] + vector_field[r_i, r_j][1] + vector_field[u_i, u_j][1] + vector_field[d_i, d_j][1] / (vector_field[l_i, l_j][2] + vector_field[r_i, r_j][2] + vector_field[u_i, u_j][2] + vector_field[d_i, d_j][2])

            vector_field[i, j] = [
                dx,
                dy,
                0
            ]
            
    for i in range(vector_field.shape[0]):
        for j in range(vector_field.shape[1]):
            den = math.sqrt(vector_field[i,j][0]**2 + vector_field[i,j][1]**2)
            if den == 0: den = 1

            # vector_field[i,j][0] /= den
            # vector_field[i,j][1] /= den

            x.append(i)
            y.append(j)
            u.append(vector_field[i,j][0])
            v.append(vector_field[i,j][1])
            

    # plot averaged vector field
    plt.quiver(x,y,u,v)
    plt.show()


    with open('vector field.pkl', 'wb') as f:
        pickle.dump([vector_field, average_acceleration], f)




