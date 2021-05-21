# Brain MRI Images for Brain Tumor Detection

This is a mini project that i took on, to learn and get hands on experience on CNNs to classify image data.

## The data set is taken from kaggle.

Link: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

## Description of what I did in this:

### Part 0 - Easier Way To Split Into Train , Test Data

1. Here i have used a library called splitfolders, which is highly useful for image datasets.
This library splits the original data into train and test folders with the specified amount of instances in both.
I have used a ratio of 70:30 

ie. 

Train : test =  0.7 : 0.3


### Part 1 - Data Preprocessing

In this step, the processing of train images will be done using ImageDataGenerator

With the following parameters:

rescale = 1./255
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True

The images will be scaled and then passed up on to next step.

The next step is flow_from_directory, which decides the input shape and the number of classes of the train data set.

I have taken input shape as (64,64,3) and class_mode= 'binary' since there are two classes, Yes and No.



### Part 2 - Building The CNN

In this step the actual CNN model is developed.

Which has 7 parts:

1. Initializing
2. Convolution where, filters=32, kernel_size=7, activation='relu', input_shape=[64,64,3]
3. Pooling where, pool_size=2, strides=2
4. Adding a second layer of convolution and pooling
5. Flattening 
6. Full connection(dense layer) where, units=128, activation='relu'
8. Output layer units=1 and activation is softmax

### Part 3 - Training The CNN and summary

In this step the model will be compiled and trained.

1. Compile where optimizer is 'adam', loss function used is 'binary_crossentropy' and metrics used is  'accuracy'
2. Summary of the model

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape                                  Param #   
-----------------------------------------------------------------
conv2d_2 (Conv2D)            (None, 58, 58, 32)                            4736      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 29, 29, 32)                             0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 27, 27, 32)                             9248      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 13, 13, 32)                             0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 5408)                                    0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)                                     692352    
_________________________________________________________________
dense_3 (Dense)              (None, 1)                                       129       
-----------------------------------------------------------------
Total params: 706,465
Trainable params: 706,465
Non-trainable params: 0
_________________________________________________________________



3. Training, where epochs are 121 and validation is done on the test set.

### Part 4 - Results

Train accuracy= 0.994 or 99.4%
Train loss= 0.034

Test accuracy= 0.857 or 85.7%
Test Loss= 

### Part 4 - Making Single Predictions

Out of 4 random samples the model predicted 4 of them correctly.
