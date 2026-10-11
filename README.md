# Gestures and Actions

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-5C3EE8?logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

2D human pose estimation with a pre-trained OpenPose model (Caffe) executed through OpenCV's DNN module. The notebook processes an input image, extracts a confidence map for each body part, locates the most probable keypoints, maps them back to the original image size and draws the skeleton.

<p align="center">
  <img width="205" height="333" alt="Input image" src="https://github.com/user-attachments/assets/06fc4779-989a-40d2-a216-3111513cf2e2" />
  &nbsp;&nbsp;
  <img width="320" alt="Detected keypoints and skeleton" src="https://github.com/user-attachments/assets/85e2ae35-4598-49a0-9fb4-fa0b9bc7ca2a" />
</p>

## How it works

1. Loads the Caffe network (`.prototxt` architecture and `.caffemodel` weights) with `cv2.dnn.readNetFromCaffe`.
2. Preprocesses the image with `cv2.dnn.blobFromImage`, normalizing pixels to the 0 to 1 range.
3. Runs a forward pass and reads the output tensor `(batch, keypoints, height, width)`.
4. For each of the 15 keypoints, takes the spatial maximum of its confidence map with `cv2.minMaxLoc`.
5. Rescales the coordinates to the original image and discards detections below the confidence threshold.
6. Connects the keypoint pairs to draw the body skeleton.

## Model

The notebook uses the **OpenPose MPI** model (15 keypoints, faster 4-stage variant) from the Carnegie Mellon University Perceptual Computing Lab:

- Architecture: [`pose_deploy_linevec_faster_4_stages.prototxt`](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt)
- Weights: `pose_iter_160000.caffemodel`, downloadable with the `getModels` script of the [OpenPose repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models)

## Tech stack

- Python 3
- OpenCV (DNN module)
- NumPy
- Matplotlib

## Repository structure

```text
Gestures-AND-Actions/
├── gestures-and-actions.ipynb   # full pipeline, step by step
├── jogadores/                   # sample images of players
└── megan/                       # additional sample image
```

## How to run

The notebook was developed on Kaggle, with the images and model files loaded as Kaggle datasets. To run it locally:

1. Install the dependencies:

   ```bash
   pip install opencv-python numpy matplotlib notebook
   ```

2. Download the model files listed above into a `model/` folder.
3. Update the image and model paths in the notebook cells that call `cv2.imread` and `cv2.dnn.readNetFromCaffe`.
4. Run `jupyter notebook gestures-and-actions.ipynb`.

## What this project shows

- Deployment of Caffe models with OpenCV DNN
- Manual interpretation of CNN confidence heatmaps
- Keypoint localization through spatial maximum
- Coordinate transformation between network output and original image

## Author

**José Isaías** · Computer Engineering at Inteli ·
[LinkedIn](https://www.linkedin.com/in/joseisaias) ·
[GitHub](https://github.com/JoseIsaias-AI)
