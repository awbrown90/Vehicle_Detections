# Created By Aaron Brown
# Udacity SDCND Project 5
# 2/15/17

# Collect all the vehicle and not-vehicle training data, extract their features and use them
# to train an SVM, then save all the SVM data into a pickle file for later.

import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import pickle
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from lesson_functions import *

# Read in car and non-car images
cars = []
notcars = []

images = glob.glob('vehicles/GTI_Far/*.png')
for image in images:
    cars.append(image)
images = glob.glob('vehicles/GTI_Left/*.png')
for image in images:
    cars.append(image)
images = glob.glob('vehicles/GTI_MiddleClose/*.png')
for image in images:
    cars.append(image)
images = glob.glob('vehicles/GTI_Right/*.png')
for image in images:
    cars.append(image)
images = glob.glob('vehicles/KITTI_extracted/*.png')
for image in images:
    cars.append(image)
images = glob.glob('non-vehicles/*.png')
for image in images:
    notcars.append(image)

print("length of cars is ",len(cars))
print("length of notcars is ",len(notcars))

# Define feature parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off   

t=time.time()
test_cars = cars
test_notcars = notcars

car_features = extract_features(test_cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = extract_features(test_notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)


print(time.time()-t, 'Seconds to compute features...')

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.1, random_state=rand_state)

print('Using:',orient,'orientations,',pix_per_cell,
    'pixels per cell,', cell_per_block,'cells per block,',
     hist_bins,'histogram bins, and', spatial_size,'spatial sampling')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
print(round(time.time()-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

svc_dict = {'svc':svc, 'scaler':X_scaler, 'orient':orient, 'pix_per_cell':pix_per_cell, 
            'cell_per_block':cell_per_block, 'spatial_size':spatial_size, 'hist_bins':hist_bins}
pickle.dump(svc_dict, open("svc_pickle.p", "wb" ) )