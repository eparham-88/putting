import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from Ball import Ball
import matplotlib.pyplot as plt


class Waypoint:

    def __init__(self, position_1, position_2, index_1, index_2):
        self.position = position_2

        self.delta = np.array([
            position_2[0] - position_1[0],
            position_2[1] - position_1[1]
        ])

        self.velocity = self.delta / float(index_2-index_1)

        self.acceleration = np.array([])

        self.index = index_2


    def find_acceleration(self, previous_waypoint):
        delta_velocity = self.velocity - previous_waypoint.velocity
        self.acceleration = delta_velocity / float(self.index - previous_waypoint.index)

    def apply_friction(self, friction_acceleration):
        # determine direction
        den = np.linalg.norm(self.delta)
        x_dir = self.delta[0] / den
        y_dir = self.delta[1] / den

        accel = 0.7*np.linalg.norm(self.acceleration)

        self.acceleration[0] -= x_dir*friction_acceleration
        self.acceleration[1] -= y_dir*friction_acceleration




class Path:

    def __init__(self, ball, stride):
        self.waypoints = []
        total_acceleration = 0.0

        # compute waypoint velocities
        for i in range(1, len(ball.positions), stride):
            waypoint = Waypoint(ball.positions[i-stride], ball.positions[i], ball.indices[i-stride], ball.indices[i])
            self.waypoints.append(waypoint)

        
        for i in range(1, len(self.waypoints)):
            self.waypoints[i].find_acceleration(self.waypoints[i-1])
            total_acceleration += np.linalg.norm(self.waypoints[i].acceleration)

        self.waypoints.pop(0)


        self.average_acceleration = total_acceleration / float(len(self.waypoints))
            



    def apply_friction(self,friction_acceleration):
        for waypoint in self.waypoints:
            waypoint.apply_friction(friction_acceleration)






if __name__ == "__main__":

    # load balls
    with open('balls.pkl', 'rb') as f:
        balls = pickle.load(f)

    # Use balls to compute paths and average friction
    total_acceleration = 0.0
    paths = []
    stride = 5

    for ball in balls:
        if len(ball.positions) < 3*stride: continue
        path = Path(ball, stride)
        if path.average_acceleration < 0.1: continue
        paths.append(path)
        print(path.average_acceleration)
        total_acceleration += path.average_acceleration
    
    friction_acceleration = 0.8 * total_acceleration / float(len(paths))
    print(friction_acceleration)

    # adjust for friction, build lists vector field
    x = []; y = []; u = []; v = []
    for path in paths:
        path.apply_friction(friction_acceleration)

        for waypoint in path.waypoints:
            u.append(waypoint.acceleration[0])
            v.append(waypoint.acceleration[1])
            x.append(waypoint.position[0])
            y.append(1280-waypoint.position[1])


    # plot raw vector field
    plt.quiver(x,y,u,v)
    plt.show()


    # channels are dx, dy, count
    cell_size = 80
    vector_field = np.zeros((int(720/cell_size), int(1280/cell_size), 3))
    x = []; y = []; u = []; v = []

    for path in paths:

        for waypoint in path.waypoints:
            x_index = min(round(waypoint.position[0]/cell_size), int(720/cell_size)-1)
            y_index = min(round(waypoint.position[1]/cell_size), int(1280/cell_size)-1)

            vector_field[x_index, y_index] += [waypoint.acceleration[0], waypoint.acceleration[1], 1]


    # fix waypoints with more than one point
    more_than_one = vector_field[:,:,2] > 1

    vector_field[more_than_one] = vector_field[more_than_one] / np.vstack((vector_field[more_than_one][:,2], vector_field[more_than_one][:,2], vector_field[more_than_one][:,2])).T

    


    # fill in nearest neighbors
    for i in range(vector_field.shape[0]):
        for j in range(vector_field.shape[1]):

            if vector_field[i, j][2] != 0: continue

            directions = [[0,1],
                          [1,0],
                          [0,-1],
                          [-1,0],
                          [1,2],
                          [-1,2],
                          [1,-2],
                          [-1,-2],
                          [1,1],
                          [-1,1],
                          [1,-1],
                          [-1,-1],
                          [2,1],
                          [-2,1],
                          [2,-1],
                          [-2,-1]]
            
            neighbors = []
            
            for direction in directions:
                new_i = i
                new_j = j
                while(new_i >= 0 and new_i <= vector_field.shape[0]-1 and new_j >= 0 and new_j <= vector_field.shape[1]-1):
                    if vector_field[new_i, new_j][2] != 0:
                        neighbors.append([
                            vector_field[new_i, new_j][0],
                            vector_field[new_i, new_j][1],
                            math.sqrt(float(new_i-i)**2 + float(new_j-j)**2)
                        ])
                        break
                    else:
                        new_i += direction[0]
                        new_j += direction[1]

            dx_total = 0; dy_total = 0; weight_total = 0
            for neighbor in neighbors:
                weight = (1/neighbor[2])
                dx_total += weight*neighbor[0]
                dy_total += weight*neighbor[1]
                weight_total += weight


            vector_field[i, j] = [
                dx_total / weight_total,
                dy_total / weight_total,
                0
            ]
            




    for i in range(vector_field.shape[0]):
        for j in range(vector_field.shape[1]):
            den = math.sqrt(vector_field[i,j][0]**2 + vector_field[i,j][1]**2)
            if den == 0: den = 1

            # vector_field[i,j][0] /= den
            # vector_field[i,j][1] /= den

            x.append(i)
            y.append(int(1280/cell_size)-j)
            u.append(vector_field[i,j][0])
            v.append(vector_field[i,j][1])
            

    # plot averaged vector field
    plt.quiver(x,y,u,v)
    plt.show()


    with open('vector field.pkl', 'wb') as f:
        pickle.dump([vector_field, friction_acceleration], f)




