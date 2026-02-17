# Gestures-AND-Actions
This project implements human body keypoint detection using a pre-trained deep learning model integrated with OpenCV’s DNN module. The system processes an input image, extracts confidence maps for each body part, identifies the most probable keypoints, and maps them back to the original image dimensions.

<img width="205" height="333" alt="image" src="https://github.com/user-attachments/assets/06fc4779-989a-40d2-a216-3111513cf2e2" />

Human Pose Estimation with OpenCV DNN (Caffe)
Overview

This project implements 2D human pose estimation using a pre-trained OpenPose COCO model (Caffe) executed through OpenCV’s DNN module.

The system:

Loads a Caffe model (.prototxt + .caffemodel)

Preprocesses the input image using blobFromImage

Performs a forward pass through the CNN

Extracts confidence heatmaps for each keypoint

Applies spatial argmax using cv2.minMaxLoc

<img width="564" height="593" alt="image" src="https://github.com/user-attachments/assets/85e2ae35-4598-49a0-9fb4-fa0b9bc7ca2a" />


Rescales coordinates to the original image size

Filters detections using a confidence threshold

This demonstrates understanding of CNN output tensors, confidence maps, and coordinate transformations.

Model Used

The project uses the OpenPose COCO model from the
Carnegie Mellon University Perceptual Computing Lab.

Download the Caffe Framework (if needed)

Caffe official repository:
https://github.com/BVLC/caffe

Download the Pre-trained Model

Prototxt (network architecture):
https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/openpose_pose_coco.prototxt

Pre-trained weights (.caffemodel):
http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel

Alternative source:
CMU-Perceptual-Computing-Lab OpenPose repository
https://github.com/CMU-Perceptual-Computing-Lab/openpose

Place both files inside a model/ directory before running the script.

What This Project Shows

Deployment of Caffe models using OpenCV DNN

Manual interpretation of CNN heatmaps

Keypoint localization via spatial maximum

Practical application of deep learning in computer vision
