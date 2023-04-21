import cv2
import math


class Ball:
    def __init__(self,position,radius):
        self.positions = [position]
        self.radii = [radius]
        self.velocities = [(0,0)]
        self.updated = True

    def update(self,ball):
        self.positions[-1] = ball.positions[-1]
        self.radii[-1] = ball.radii[-1]
        self.updated = True

    def velocity_from_previous(self,ball):
        (x0, y0) = self.positions[-1]
        (x1, y1) = ball.positions[-1]
        r0 = self.radii[-1]
        r1 = ball.radii[-1]

        # compute normalized ball speed. normalize by averaging both radii
        dx = x1 - x0 # / (r0 + r1) 
        dy = y1 - y0 # / (r0 + r1)

        self.positions.append(ball.positions[-1])
        self.radii.append(ball.radii[-1])
        self.velocities.append((dx, dy))
        self.updated = True

        return (dx, dy)
    
    def display(self,frame):

        cutoff = 15

        

        if len(self.positions) == cutoff:
            # check validity
            dx = self.positions[-1][0] - self.positions[0][0]
            dy = self.positions[-1][1] - self.positions[0][1]

            den = math.sqrt(dx**2 + dy**2)
            if den == 0: den = 1

            dx = dx / den
            dy = dy / den
            
            deviation = 0
            for i in range(1, len(self.positions)):
                this_dx = self.positions[i][0] - self.positions[0][0]
                this_dy = self.positions[i][1] - self.positions[0][1]

                den = math.sqrt(this_dx**2 + this_dy**2)
                if den == 0: den =1
                this_dx = this_dx / den
                this_dy = this_dy / den

                deviation += abs(dx-this_dx) + abs(dy-this_dy)

            print(deviation)
            if deviation > 5:
                self.positions = [self.positions[-1]]

            # draw every line up to and including the cutoff
            for i in range(1, len(self.positions)):
                start = (int(self.positions[i-1][0]), int(self.positions[i-1][1]))
                end   = (int(self.positions[i][0]), int(self.positions[i][1]))
                width = int(0.5 * (self.radii[i-1] + self.radii[i]))

                cv2.line(frame, start, end, (255, 0, 0), width)

        elif len(self.positions) > cutoff:
            # just append next line
            start = (int(self.positions[-2][0]), int(self.positions[-2][1]))
            end   = (int(self.positions[-1][0]), int(self.positions[-1][1]))
            width = int(0.5 * (self.radii[-1] + self.radii[-2]))

            cv2.line(frame, start, end, (255, 0, 0), width)


    
def ball_distance(ball1, ball2):
    return math.sqrt( (ball1.positions[-1][0] - ball2.positions[-1][0])**2 +
                      (ball1.positions[-1][1] - ball2.positions[-1][1])**2 )

def direction_matches(ball1, ball2):
    ball1_dx = ball1.positions[-1][0] - ball1.positions[0][0]
    ball1_dy = ball1.positions[-1][1] - ball1.positions[0][1]

    if math.sqrt(ball1_dx**2 + ball1_dy**2) < 30: return True 

    proposed_dx = ball2.positions[-1][0] - ball1.positions[-1][0]
    proposed_dy = ball2.positions[-1][1] - ball1.positions[-1][1]

    theta1 = math.atan2(ball1_dy,ball1_dx)
    proposed_theta = math.atan2(proposed_dy,proposed_dx)

    return abs( theta1 - proposed_theta ) < 0.2