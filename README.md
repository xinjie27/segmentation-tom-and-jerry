# Instance Segmentation on Tom and Jerry
16 December, 2021

## Project Description
This is a starter project for Brown Visual Computing Group. In this project, I implemented a U-Net model using TensorFlow 2 for image segmentation tasks. The data were collected from the famous cartoon *Tom and Jerry*, and the task was to locate and partition the Jerry Mouse in selected video frames. By training on 80 images and their true masks, the U-Net model has achieved an Intersection-Over-Union score of 0.8, significantly higher than random guessing.

## Dataset
The dataset consists of 80 training images and 10 test images. It was generated from *Tom and Jerry* Episode 24, *The Milky Waif* (1946). Video frames were extracted and selected so that Jerry Mouse appeared clearly and completely, not obscured by other objects.

I used [Labelme](https://github.com/wkentaro/labelme) for image annotations. It is a graphical annotation tool written in Python, and we can label the border of target instance . In instance segmentation tasks, it could highlight the boarder by polygons

<img src="readme_imgs\dataset_1.png" alt="Image Annotation" width="400" height="300" />

After annotating the data,

## Model

## Usage

## Results and Visualizations