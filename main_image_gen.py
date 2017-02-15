# Created By Aaron Brown
# Udacity Self-Driving Car Nanodegree Project 5
# Feburary 10, 2017

# Main class to generate image output

import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from skimage.feature import hog
import os, os.path
import pickle
from extra_functions import *

# Load up saved data from SVM post training
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

example_images = glob.glob('./test_images/test*.jpg')

# Define the y range to search for cars
ystart = 400
ystop = 656
# Iterate over test images
for idx, img_src in enumerate(example_images):

	print("working on file ",idx)
	img = mpimg.imread(img_src)
	# Create a heatmap template
	heatmap = np.zeros_like(img[:,:,0])
	# Iterate through different scale values
	for scale in np.arange(1,2.1,.2):
		#print(scale)
		# Create heat maps for different scales in function to both search and classify
		out_img, heat_map = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
		heatmap += heat_map
	# Threshold the heatmap to get rid of false positives
	heatmap = apply_threshold(heatmap,3)
	labels = label(heatmap)
	# Draw bounding boxes on a copy of the image
	draw_img = draw_labeled_bboxes(np.copy(img), labels)

	# Fix cv2 image output
	r,g,b = cv2.split(draw_img)
	draw_img = cv2.merge((b,g,r))

	# Add some color to the heat map and scale it
	zero_channel = np.zeros_like(heatmap)
	heatmap = heatmap*20
	heatmap = cv2.merge((zero_channel,zero_channel,heatmap))

	# Save the output for car box locations and heatmaps
	write_name = './output_images/tracked'+str(idx)+'.jpg'
	cv2.imwrite(write_name, draw_img)

	write_name = './output_images/heatmap'+str(idx)+'.jpg'
	cv2.imwrite(write_name, heatmap)
