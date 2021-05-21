import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

tf.__version__

! pip install split-folders

## Easier way to split into train , test data

import splitfolders

# train, test split
splitfolders.ratio('../input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/', output="./brain_tumor_dataset_split", ratio=(0.7, 0.3))

## Part 1 - Data Preprocessing

### Preprocessing the Training set

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('./brain_tumor_dataset_split/train',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

training_set.class_indices

yes=np.count_nonzero(training_set.classes)
print("Yes:",yes)
print("No:",176-yes)

### Preprocessing the Test set

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('brain_tumor_dataset_split/val/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

yes_test=np.count_nonzero(test_set.classes)
print("Yes:",yes_test)
print("No:",77-yes_test)

## Part 2 - Building the CNN

### Initialising the CNN

cnn = tf.keras.models.Sequential()

### Step 1 - Convolution

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=7, activation='relu', input_shape=[64,64,3]))

### Step 2 - Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

### Adding a second convolutional layer

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

### Step 3 - Flattening

cnn.add(tf.keras.layers.Flatten())

### Step 4 - Full Connection

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

### Step 5 - Output Layer

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

## Part 3 - Training the CNN

### Compiling the CNN

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

### Summary of the model

cnn.summary()

### Training the CNN on the Training set and evaluating it on the Test set

cnn.fit(x = training_set, epochs = 121 ,validation_data=test_set)

### Train accuracy

cnn.evaluate(training_set)

### Test accuracy 

cnn.evaluate(test_set)

## Part 4 - Making single predictions

import numpy as np
from tensorflow.keras.preprocessing import image


from IPython.display import display, Image
display(Image(filename='./brain_tumor_dataset_split/val/no/No22.jpg'))
test_image = image.load_img('./brain_tumor_dataset_split/val/no/No22.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
if result[0][0] == 1:
  prediction = 'Yes'
else:
  prediction = 'No'

print("Actual: No")
print("Predicted:", prediction)

import numpy as np
from tensorflow.keras.preprocessing import image


from IPython.display import display, Image
display(Image(filename='./brain_tumor_dataset_split/val/no/3 no.jpg'))
test_image = image.load_img('./brain_tumor_dataset_split/val/no/39 no.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'Yes'
else:
  prediction = 'No'

print("Actual: No")
print("Predicted:", prediction)

import numpy as np
from tensorflow.keras.preprocessing import image


from IPython.display import display, Image
display(Image(filename='./brain_tumor_dataset_split/val/yes/Y147.JPG'))
test_image = image.load_img('./brain_tumor_dataset_split/val/yes/Y147.JPG', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'Yes'
else:
  prediction = 'No'

print("Actual: Yes")
print("Predicted:", prediction)

import numpy as np
from tensorflow.keras.preprocessing import image


from IPython.display import display, Image
display(Image(filename='./brain_tumor_dataset_split/val/yes/Y3.jpg'))
test_image = image.load_img('./brain_tumor_dataset_split/val/yes/Y3.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'Yes'
else:
  prediction = 'No'

print("Actual: Yes")
print("Predicted:", prediction)