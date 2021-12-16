# Instance Segmentation on Tom and Jerry
16 December, 2021



## Project Description
This is a starter project for Brown Visual Computing Group. In this project, I implement a U-Net model using TensorFlow 2 for image segmentation tasks. The data are collected from the famous cartoon *Tom and Jerry*, and the task is to locate and partition the Jerry Mouse in selected video frames. By training on 80 images and their true masks, the U-Net model has achieved an Intersection-Over-Union score of 0.8, significantly higher than random guessing.



## Data
#### The Dataset

The dataset consists of 80 training images and 10 test images. It is generated from *Tom and Jerry* Episode 24, *The Milky Waif* (1946). Video frames are extracted and selected so that the Jerry Mouse appears clearly and completely, not obscured by other objects. The data are zipped and stored in ```data.zip```.

#### Data Annotations

Since image segmentation is a pixelwise classification task, we need to label every pixel of our data. In this project, I used [Labelme](https://github.com/wkentaro/labelme) for pixelwise image annotations. It is a graphical annotation tool written in Python, which can automatically generate pixel-level ground truth masks after we outline the border of our target instance, Jerry Mouse, in polygons. Here is an example:

<img src="readme_imgs\1_image_annotation.png" alt="Image Annotation" width="400" height="300" />

Image annotations are stored in JSON format, and one can convert them by running ```labelme2voc.py```. This script creates 4 directories: the first two directories correspond to our input images (stored in ```.jpg``` format) and masks (stored in ```.npy``` format), and the other two directories are visualizations.



## Model

#### The U-Net Architecture

The U-Net consists of a contracting path to capture context and a symmetric expanding path that enables precise localization ([see the original paper](https://arxiv.org/abs/1505.04597)). Its architecture is illustrated below:

<img src="readme_imgs\2_u-net.png" alt="U-Net Architecture" width="400" />

In ```model.py```, I implement the U-Net model by three ```tf.keras.Model``` classes: ```ConvBlock()```, ```EncoderBlock()```, and ```DecoderBlock()```.

+ The ```ConvBlock()``` contains two 3x3 convolution layers, each followed by batch normalization and ReLU activation function; it is an element of both the encoder and the decoder.
+ The ```EncoderBlock()``` is one of the five steps in the contracting (left) path shown above; it consists of a ```ConvBlock()```and a 2D max pooling layer.
+ The ```DecoderBlock()``` is one of the five steps in the expanding (right) path shown above; it consists of a transposed convolution layer (upsampling), a skip connection process (the grey "copy and crop" arrows), and a ```ConvBlock()```. 

Finally, the ```UNet()``` class is constructed by an input layer, an entire contracting path and an entire expanding path.

#### Loss Function

There are two output classes: 1 (Jerry Mouse) and 0 (not the Jerry Mouse). Therefore, we use binary cross entropy as the loss function. However, that leads to a problem: Jerry Mouse is a small instance compared to the background, so in our ground truth masks, the majority of pixels are of class 0. In this way, if we use the vanilla binary cross entropy, our optimizer might learn a local optimum where all pixels are classified to 0! To fix this problem, we not only need to fine tune our hyperparameters, but more importantly, assign different weights to our output classes. This is done by writing a custom binary cross entropy loss function, and the default weight ratio between class 1 and 0 is 5 to 1.



## Usage

Run the script ```main.py``` to access the project. Data are unzipped and stored in ```./data``` directory, and an output directory (```./outputs``` by default) is set to save the model weights. There are two modes to run the program:

#### Train

```shell
python main.py -m train -i ./data/train -o ./outputs
# Arguments for hyperparameter tuning:
# -e or --epochs <---> Number of epochs
# -b or --batch-size <---> Batch size
# -l or --learning-rate <---> Learning rate
# --class-weight <---> Class weight (for loss function)
python main.py -m train -i ./data/train -o ./outputs -e 5 -b 2 -l 3e-5 --class-weight 5
```

#### Test

```shell
python main.py -m test -i ./data/test -o ./outputs
```



## Results

#### Random Guessing Baseline

We first instantiate a ```UNet()``` model using random weights, and run the model on a test set of 10 images. The mean IoU (Intersection-Over-Union) score is **0.066**. Here is a visualization of a test image with its true mask and predicted mask, respectively:

<img src="readme_imgs\3_baseline.png" alt="Baseline" height="250" />

One can see that our random guessing model performs badly: it can neither detect nor partition the Jerry Mouse.

#### Model Performance

We then train our U-Net model on 80 images, and the model has achieved an IoU score of **0.815** on the test set. This is significantly higher than our baseline performance. Here is a visualization of the same test image, with its true mask and predicted mask:

<img src="readme_imgs\4_model_performance.png" alt="Model Performance" height="250" />

Apparently, the model can now detect and locate the Jerry Mouse! The border is not precise though, and this is primarily because we have a very limited amount of data. I have already implemented the data augmentation technique, including randomly flipping the image horizontally and vertically, but if we have a much larger dataset (let's say 1000 training images), the model can surely improve greatly.