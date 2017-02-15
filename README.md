# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Check out the [Video Results](https://www.youtube.com/watch?v=yWY3Y4pbDxs&feature=youtu.be)

In this project, the goal was to write a software pipeline to detect vehicles in a video, frame by frame. In order to do this a Support Vector Machine Classifer was used and trained on thousands of images of car and non-cars. The features of the SVC consisted of hog, spatial color binning, and color historgrams. Once the SVC had a high accuracy of detecting car images from non-car images, 99% final accuracy, a vehicle class and Sync pipeline was introduced to keep track of new captures from old captures and properly update the tracked output. 

## Training and Feature Extraction

The training set consisted of a total of about 9,000 images each for car and non-car examples, and each image was 64 x 64 pixels with 3 color channels. Hog is a feature extraction method that consists of breaking the image up into cells and then measuring the strongest graident direction in each cell. By doing this information about the shape, indepent of the color can be captured. The cell size used for calculating hog was 8 x 8 cells where each cell was then 8 x 8 pixels. Further more the orientation of each gradient was grouped into one of 9 different resoultion bins. Finally each cell with its orientation bin was unraveled per color channel and stored into a 1-demensional feature vector. 

Complementing the hog feature vector was then the spatial color binning that just consisted of a down sampled input image of 32 x 32 pixels and unraveled into a 1-dementional vector as well. Then the color historgram used 32 different resoultion bins for each color channel. The color space used in this project however was not RGB but instead YCrCb which proved to have very useful results when extracting features. Since all of these feature vectors were 1-dimensional they could all simply be stacked ontop of each other for a final feature vector representation.

[YCrCb example](https://www.google.com/search?q=YCbCr&rlz=1C1CHBD_enUS702US702&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiMgpmdnpPSAhXDLmMKHaleBOcQ_AUICCgB&biw=1536&bih=806#tbm=isch&q=ycbcr+vs+rgb&imgrc=OQhfRMzOkJo7GM:)

With the following described feature vector values and using the large training dataset, the SVC was able to achieve 99% classification accuracy with a test set that was 10% of the orginal training data. This high accuracy was very important for the detection pipeline in order to minimize the number of false postive and negative detections.

## Pipeline with test images

Once the SVC was working well further techniques were used in order to group detected results and minimize the number of false positives. Introducing the idea of a heatmap was an effective way of overlaying multi-scaled window resutls. By breaking the test image into mulitple lists of scaled windows the SVC could make a prediction on each one and then add car predicted region areas onto an overall heatmap image. The more overlapping window areas that the SVC identifed as a car, the larger the values the heatmap would have in that area. 

Its important to note that inorder to improve the efficeny of extracting features from test image windows that a hog sub-sampling method was used that could calculate a hog features only once on a defined image height range and then using a scaling based on the 8 x 8 pixel cells eariler defined to slide and search for windows. so for instance the smallest search window you could have then is 64 x 64 pixels and then larger windows could be made by extending 8 pixel cells to either x or y dimensions of the window. The spatial and histrograms features would still be extracted per window by resizing the target window to 64 x 64 pixels. In this project 5 different scaled search windows were used that went from scale of 1x to 2x in increments of .2 so windows of 64x64 to 128 x 128 pixels. 

Once the final heatmaps were created by using the search classify function discussed above, the heatmap was thresholded so only areas with a minimum number of overlaps would be considered for detections. This considerably helped removing false positives from the heatmap. The heatmap was then given to the lables function which grouped nearby areas into different boxed detections which could be drawn onto the orginal image showing a detected cars area.

## Pipeline with videos

The video pipeline used all the same methods as the image pipeline but also used a vehicle class and a sync function. A vehcile class was useful for storing all a detected car's position and window history values, and averaging those as an output to smooth recorded results when displayed on screen. The sync function was used to match new captures to old captures by using a capture's position as a reference. When a new capture was detected that did not match with an old capture, that new caputure was assigned a new vehicle class and added to the tracked carslist. When an old capture was not matched with a new capture, that vehicle was demoted, if it was demoted 3 times in a row then it would be deleted from the tracked carslist. Likewise a vehicle needed to be detected 5 times in a row before it could be displayed. 

Also for videos, heatmaps were created by considering a series of consecutive frames and adding together each indivdual frames heatmap together, then thresholding that final resutl by a value of 5. This technuiqe proved to give good results, and the end result was identifying all the main cars in the project video, the cars driving in the same direction as the camera car, and not having any false positives, the only other minor detections came from cars driving in the opposite direction. The tracked bounding boxes around cars in videos looked smooth because the vehicle class was returning positions and window width/height values averaged over the last 5 frames.

The next steps for this project is to combine it with the adavance lane finding pipeline and try it on the other project videos as well as maybe some external videos to get a full idea of how well the pipeline really works.

