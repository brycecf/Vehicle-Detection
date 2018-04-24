# **Advanced Lane Finding**

## Project Goal

---

The goal of this project was to create a machine learning-based pipeline for detecting vehicles in video streams.

[//]: # (Image References)

[orig-cars]: ./output_images/orig_cars.png
[hog-cars]: ./output_images/hog_cars.png
[orig-noncars]: ./output_images/orig_noncars.png
[hog-noncars]: ./output_images/hog_noncars.png
[heatmap-imgs]: ./output_images/heatmap_imgs.png

---

## Histogram of Oriented Gradients (HOG)

I largely relied on `skimage.feature.hog` to extract HOG features from the training images (both vehicle and non-vehicle ones). Using this algorithm, I explored each of the possible color spaces (with a particular focus on RGB and HSV). 

Ultimately, I used YCrCb due to its role within video recordings. Through a trial-and-error approach, I arrived at the following parameters for the HOG extraction:

* `spatial_size` = (32, 32)
* `orient` = 12
* `pix_per_cell` = 8
* `cell_per_block` = 2
* `hist_bins` = 32

#### HOG Features for Vehicles

Let us look at a sample of the original vehicle images below.

![alt text][orig-cars]


Now, here are the HOG features for those same images.

![alt text][hog-cars]

Looking at the HOG features, it acts like an early layer in a convolutional neural network. We can see how it highlights edges on the vehicle, as well as windows.

#### HOG Features for Non-Vehicles

Next, let us take a look at a sample of the original non-vehicle images.

![alt text][orig-noncars]

Here are the associated HOG features.

![alt text][hog-noncars]

Since the HOG features are acting like a shape/edge detector, we can clearly see a difference between how non-vehicles and vehicles are represented.

Besides the HOG features, I also used spatial binning and color histograms as additional features since these provided additional information on color details, which would distinguish a vehicle on the road.

This code was implemented in the fourth code block within `extract_features()`.

---

## Vehicle Classification Model

A support vector classifier with a linear kernel was used to determine if an image was of a vehicle or a non-vehicle. I performed an 80/20 stratified split on a dataset of 17,760 images, which resulted in a training set of 14,208 images and 3,552 test images. Since there was a roughly equal number of vehicle and non-vehicle images, the stratified split was effective on its own.

Utilizing the default hyperparameters, the model was able to achieve an F1-score of ~0.992 on the test set.

This was done in the 13th-21st code blocks in the notebook.

---

## Sliding Window Search

Like other object detection applications, a sliding window approach was used to search the video frames for vehicles. This was largely implemented in the second code block within `find_cars()` and the 22nd code block under `process_img()`. Like the feature hyperparameters before, I used a trial-and-error approach to select the hyperparameters for this algorithm. A 64x64 window with a standard scale and a 1.5 scale were used based on the scale of the vehicles within the image and its performance in the class tutorials. The scale was selected since it was able to effectively capture vehicles without greatly impacting the computational time. Furthermore, a heatmap was used to identify false positives.

---

## Pipeline Demonstration Images & Heatmap Examples

Below are images demonstrating the pipeline's performance on the sample test images, as well as demonstrating the heatmap's functionality (for detecting false positives).

![alt text][heatmap-imgs]

Through a process of trial-and-error, I evaluated the pipeline's performance using different settings until I ultimately decided upon the three concatenated features (described earlier) and two scales.

---

## Pipeline Video Demonstration

A video demonstration of the vehicle detection pipeline can be found [here](https://www.youtube.com/watch?v=wUwak2uNrG0&feature=youtu.be).

---

## False Positive Filtering and Overlapping Bounding Boxes

As mentioned earlier, I utiliezd heatmaps to detect false positives. The heatmap code can be found in the second code block in `find_cars()` and the heatmap thresholding is in the 22nd code block in `process_image()`. As recommended, I used `scipy.ndimage.measurements.label()` to detect the "hot zones" within the heatmap. Once those zones were identified, a bounding box was placed around the "hot zone" to capture a vehicle's location.

---

## Discussion

The pipeline will likely fail in low-visibility conditions due to its reliance on color gradients and histograms. This also poses challenges in areas with city lights for example.

In general, as in the previous project, a neural network (and deep learning approach) would likely improve the robustness of the model through its automated feature extraction.