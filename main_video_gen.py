# Created By Aaron Brown
# Udacity Self-Driving Car Nanodegree Project 5
# Feburary 10, 2017

# Main class to generate video output

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
from vehicle import vehicle
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

# Define the y range to search for cars
ystart = 400
ystop = 656
# Save off heatmaps and car detected
heatmaps = []
carslist = []

# Function used to sync new detections with existing saved car detections
def Sync(captured, carlist):


	#print('used for debugging: start sync')
	#print('caps ',len(captured))
	#print('cars ',len(carlist))

	# If no cars add any detections as a new vehicle and put in carlist
	if len(carlist) == 0:
		for capture in captured:
			carlist.append(vehicle(capture[0],capture[1],capture[2],capture[3]))

	# If no captures but carlist is not empty then demote all vehicle detections
	elif len(captured) == 0:
		for car in carlist:
			if car.NegUpdate():
				carlist.remove(car)

	# Both carlist and new detections are non-empty and need to do comparisons to pair results
	else:
		# Keep track of what captures need to be made into a new vehicle
		capture_update = np.zeros((len(captured)))
		# Keep track of what cars need to get demoted because they didnt have a new track
		remove_update = np.zeros((len(carlist)))
		# Create a table to keep track of how new captures and existing captures relate
		# new captures are stored on the rows and exisitng captures on the columbs
		# The values we are storing between a new/old caputre pair is the rank defined as distance between positions
		grid = np.zeros((len(captured),len(carlist)+1))

		# Iterate through new captures
		for x in range(len(captured)):
			# Iterate through old captures
			for o in range(len(carlist)):
				# Measure the distance between new capture and all old captures
				grid[x,o] = carlist[o].Rank(captured[x][0],captured[x][1])
			# Store the old capture that was closest to the new capture 
			grid[x,len(carlist)] = np.argmin(grid[x][:-1])
		# Iterate through old captures
		for o in range(len(carlist)):
			# Iterate through new captures
			for x in range(len(captured)):
				# Mark all ranks as very high if new/old capture pair didnt match
				if grid[x][-1] != o:
					grid[x,o] = 1000

			# If the rank is good between new/old capture pair
			if np.amin(grid[:,o]) < 1000:
				# Establish a link to new/old capture and update new detected positon and window dimensions
				get_capture = captured[np.argmin(grid[:,o])]
				carlist[o].PosUpdate(get_capture[0],get_capture[1],get_capture[2],get_capture[3])
				capture_update[np.argmin(grid[:,o])] = 1
			else:
				# Else old capture did not have a new capture
				remove_update[o] = 1
		# Check what new captures did not have a matched old capture and create a new vehicle out of it, and add it in carlist
		for i in range(len(capture_update)):
			if capture_update[i] == 0:
				get_capture = captured[i]
				carlist.append(vehicle(get_capture[0],get_capture[1],get_capture[2],get_capture[3]))

		# Look through old captures that did not have a new capture and demote them, if the negtative detections in a row is
		# greater than some threshold we delete the old capture from the carlist
		remove_index = []
		for car in range(len(remove_update)):
			if remove_update[car] == 1 and carlist[car].NegUpdate():
				remove_index.append(car)

		carlist = np.delete(carlist,remove_index)
		carlist = carlist.tolist()

	return carlist

# Process each frame of the video
def process_image(image):
	# We want to modify carslist and heatmaps from this function
	global carslist, heatmaps
	# Create a general heatmap
	heatmap = np.zeros_like(image[:,:,0])
	# Iterate through different scale values
	for scale in np.arange(1,2.1,.2):
		# Create heat maps for different scales in function to both search and classify
		out_img, heat_map = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
		# Add heat to heatmap from different scales
		heatmap += heat_map
	
	# Add general heatmap to list of heatmaps
	heatmaps.append(heatmap)
	# Sum togehter the last 5 frame heatmaps into 1 heatmap
	heatmap = seriesHeatmap(image,5)
	# Use threshold to help remove false positives from heatmap
	heatmap = apply_threshold(heatmap,5)
	# Get the box positions from heatmap
	labels = label(heatmap)

	# Store box parameters for each box as a new capture
	captured = []
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		xpos = np.min(nonzerox)
		ypos = np.min(nonzeroy)

		width = np.max(nonzerox)-xpos
		height = np.max(nonzeroy)-ypos

		captured.append([xpos,ypos,width,height])

	# Sync old captures with new captures
	carslist = Sync(captured,carslist)

	# Draw bounding boxes on a copy of the image
	draw_img = draw_labeled_bboxes(np.copy(image), carslist)

	return draw_img

def draw_labeled_bboxes(img, carlist):
    # Iterate through all detected cars
    for car_number in range(len(carlist)):

    	if carlist[car_number].detected:
    		bbox = carlist[car_number].Box()
    		# Draw the box on the image
    		cv2.rectangle(img, bbox[0], bbox[1], (38, 133, 197), 6) 
    # Return the image
    return img
		
def seriesHeatmap(refimg,nframe):
	heatmap = np.zeros_like(refimg[:,:,0])
	nframe = min(5,len(heatmaps))
	for i in np.arange(1,nframe):
		heatmap += heatmaps[-i]
	return heatmap


video_output = 'tracked2.mp4'
Input_video = 'project_video.mp4'
clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
video_clip.write_videofile(video_output, audio=False)