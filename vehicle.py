# Created By Aaron Brown
# Udacity Self-Driving Car Nanodegree Project 5
# Feburary 10, 2017

# Main class to store vehicle data, such as tracking box position and width/height, history of vehicle detections
# and rank function used to measure distance between current vehicle detection and new potential detection.

import math
import numpy as np

class vehicle():
    def __init__(self, x_position, y_position, width, height, smooth_factor=5):
        self.detected = False  # is vehicle active, only draw vehicle if active
         # Store all vehciles history of x,y positions and width/height
        self.x_positions = []
        self.x_positions.append(x_position)
        self.y_positions = []
        self.y_positions.append(y_position)
        self.width = []
        self.width.append(width)
        self.height = []
        self.height.append(height)
        # Define a smoothing factor to average past x,y positions and width/height
        self.smooth_factor = smooth_factor

        self.n_detections = 1 # Number of consecutive times this vehicle has been detected?
        self.n_nondetections = 0 # Number of consecutive times this car has not been detected since last detection
    
    # Return the distance between the vehicles position and some input capture's position
    def Rank(self, x_position, y_position):
        rank = math.sqrt((x_position-self.x_positions[-1])**2+(y_position-self.y_positions[-1])**2)
        # threshold the max distance to consider a link
        if rank < 50: 
            return rank
        else:
            return 1000

    # Update the vehicle with new pos/dim
    def PosUpdate(self, x_position, y_position, width, height):
        self.x_positions.append(x_position)
        self.y_positions.append(y_position)
        self.width.append(width)
        self.height.append(height)

        self.n_detections+=1
        self.n_nondetections = 0
        if self.n_detections > 5:
            self.detected = True

    # Demote the vehicle because it didnt have a new caputre
    def NegUpdate(self): #return weather to remove vehicle
        self.n_nondetections+=1

        # If we dont see the vehicle for three frames in a row we delete it
        if self.n_nondetections > 3:
            return True
        return False

    # Check if vehicle was detected but had failed to be tracked 3 times in a row
    def CheckValid(self):
        if self.detected & self.n_nondetections > 3:
            return False
        else:
            return True

    # Average together all pos/dim and return the box that defines
    def Box(self):
        xpos = int(np.average(self.x_positions[-self.smooth_factor:], axis = 0))
        ypos = int(np.average(self.y_positions[-self.smooth_factor:], axis = 0))
        width = int(np.average(self.width[-self.smooth_factor:], axis = 0))
        height = int(np.average(self.height[-self.smooth_factor:], axis = 0))

        return ((xpos, ypos), (xpos+width, ypos+height))




