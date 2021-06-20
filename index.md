# X-ray Image Classification and Model Evaluation
## Pneumonia detection from chest X-ray images using Tensorflow
## Author: Md Sohel Mahmood
## 06/20/2021

![GitHub Logo](/images/owen-beard-DK8jXx1B-1c-unsplash.jpg)


Kaggle has a wonderful source of chest X-ray image datasets for pneumonia and normal cases. There are significant differences between the image of a normal X-ray and an affected X-ray. Machine learning can play a pivotal role in determining the disease and significantly boost the diagnosis time as well as reduce human effort. In this article, I will walk through this dataset and classify the images with an evaluation accuracy of 90%.

![GitHub Logo](/images/im1.png)


I have been motivated by the work done here (https://goodboychan.github.io/python/deep_learning/tensorflow-keras/vision/2020/10/16/01-Image-Classification-with-Cat-and-Dog.html) on the datasets between cats and dogs and reused the code block for dataset pipeline. First we need to import the necessary packages.
```
import os
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow import keras
import matplotlib.pyplot as plt
```
The initial dataset is too big. I have chosen a reduced set having 1000 images for normal case and 1000 images for pneumonia. There will be three directories: train, validation and test. Validation and test datasets will be completely new to the model and we don’t need to perform any image augmentation technique to these datasets.
```
base_dir = 'chest_xray/reduced size/'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')

train_NORMAL_dir = os.path.join(train_dir, 'NORMAL')
train_PNEUMONIA_dir = os.path.join(train_dir, 'PNEUMONIA')

validation_NORMAL_dir = os.path.join(validation_dir, 'NORMAL')
validation_PNEUMONIA_dir = os.path.join(validation_dir, 'PNEUMONIA')

train_NORMAL_fnames = os.listdir( train_NORMAL_dir )
train_PNEUMONIA_fnames = os.listdir( train_PNEUMONIA_dir )

print(train_NORMAL_fnames[:10])
print(train_PNEUMONIA_fnames[:10])

print('total training NORMAL images :', len(os.listdir(      train_cats_dir ) ))
print('total training PNEUMONIA images :', len(os.listdir(      train_dogs_dir ) ))

print('total validation NORMAL images :', len(os.listdir( validation_cats_dir ) ))
print('total validation PNEUMONIA images :', len(os.listdir( validation_dogs_dir ) ))
```
Next few samples are imported. The output shows the X-ray images from the targeted two cases.
```
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4
pic_index = 0 # Index for iterating over images

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_NORMAL_pix = [os.path.join(train_NORMAL_dir, fname)
                for fname in train_NORMAL_fnames[ pic_index-8:pic_index]
               ]

next_PNEUMONIA_pix = [os.path.join(train_PNEUMONIA_dir, fname)
                for fname in train_PNEUMONIA_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_NORMAL_pix+next_PNEUMONIA_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')

  img = mpimg.imread(img_path)
  plt.imshow(img)
```

We will assign labels to the datasets: 0 for NORMAL and 1 for PNEUMONIA. That’s why we will use SparseCategoricalCrossentropy() loss function in the model compilation. The model is defined as below:

```
class Conv(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(Conv, self).__init__()
        
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        
    def call(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.pool(x)
        return x
    
model = tf.keras.Sequential(name='X-ray_CNN')

model.add(Conv(filters=32, kernel_size=(3, 3)))
model.add(Conv(filters=64, kernel_size=(3, 3)))
model.add(Conv(filters=128, kernel_size=(3, 3)))
model.add(Conv(filters=128, kernel_size=(3, 3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

from glob import glob

base_dir = 'chest_xray/reduced size/'
train_len = len(glob(os.path.join(base_dir, 'train', 'NORMAL', '*.jpeg'))) * 2
val_len = len(glob(os.path.join(base_dir, 'val', 'NORMAL', '*.jpeg'))) * 2
test_len = len(glob(os.path.join(base_dir, 'test', 'NORMAL', '*.jpeg'))) * 2
train_len
```

Since there are equal number of images in each of the target folders, the total train or validation or test length is obtained by multiplying one folders length by 2. Next part is the image augmentation
```
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( rescale = 1.0/255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
                                    )
validation_datagen  = ImageDataGenerator( rescale = 1.0/255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
                                    )
```

We will flow training images in batches of 20 using train_datagen and validation_datagen generator.
```
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                                                 save_weights_only=False,
                                                 verbose=1)
```

A bunch of functions to manipulate the images are placed.
```
import numpy as np

def load(f, label):
    # load the file into tensor
    image = tf.io.read_file(f)
    # Decode it to JPEG format
    image = tf.image.decode_jpeg(image)
    # Convert it to tf.float32
    image = tf.cast(image, tf.float32)
    
    return image, label

def resize(input_image, size):
    return tf.image.resize(input_image, size)

def random_crop(input_image):
    return tf.image.random_crop(input_image, size=[150, 150, 1])

def central_crop(input_image):
    image = resize(input_image, [176, 176])
    return tf.image.central_crop(image, central_fraction=0.84)

def random_rotation(input_image):
    angles = np.random.randint(0, 3, 1)
    return tf.image.rot90(input_image, k=angles[0])

def random_jitter(input_image):
    # Resize it to 176 x 176 x 3
    image = resize(input_image, [176, 176])
    # Randomly Crop to 150 x 150 x 3
    image = random_crop(image)
    # Randomly rotation
    image = random_rotation(image)
    # Randomly mirroring
    image = tf.image.random_flip_left_right(image)
    return image

def normalize(input_image):
    mid = (tf.reduce_max(input_image) + tf.reduce_min(input_image)) / 2
    input_image = input_image / mid - 1
    return input_image

def load_image_train(image_file, label):
    image, label = load(image_file, label)
    image = random_jitter(image)
    image = normalize(image)
    return image, label

def load_image_val(image_file, label):
    image, label = load(image_file, label)
    image = central_crop(image)
    image = normalize(image)
    return image, label
```

These functions will induce some random jittering and help to load images. The next block will assign the labels to the individual folders. As mentioned earlier, 0 will be assigned for NORMAL and 1 will be assigned for PNEUMONIA. This is applied for all the three folders of train, validation and test.
```
temp_ds = tf.data.Dataset.list_files(os.path.join("chest_xray/reduced size", 'train', 'NORMAL', '*.jpeg'))
temp_ds = temp_ds.map(lambda x: (x, 0))

temp2_ds = tf.data.Dataset.list_files(os.path.join("chest_xray/reduced size", 'train', 'PNEUMONIA', '*.jpeg'))
temp2_ds = temp2_ds.map(lambda x: (x, 1))
train_ds = temp_ds.concatenate(temp2_ds)

buffer_size = tf.data.experimental.cardinality(train_ds).numpy()
train_ds = train_ds.shuffle(buffer_size)\
                   .map(load_image_train, num_parallel_calls=16)\
                   .batch(20)\
                   .repeat()

temp_ds = tf.data.Dataset.list_files(os.path.join("chest_xray/reduced size", 'val', 'NORMAL', '*.jpeg'))
temp_ds = temp_ds.map(lambda x: (x, 0))

temp2_ds = tf.data.Dataset.list_files(os.path.join("chest_xray/reduced size", 'val', 'PNEUMONIA', '*.jpeg'))
temp2_ds = temp2_ds.map(lambda x: (x, 1))

val_ds = temp_ds.concatenate(temp2_ds)

val_ds = val_ds.map(load_image_val, num_parallel_calls=16)\
               .batch(20)\
               .repeat()

temp_ds = tf.data.Dataset.list_files(os.path.join("chest_xray/reduced size", 'test', 'NORMAL', '*.jpeg'))
temp_ds = temp_ds.map(lambda x: (x, 0))

temp2_ds = tf.data.Dataset.list_files(os.path.join("chest_xray/reduced size", 'test', 'PNEUMONIA', '*.jpeg'))
temp2_ds = temp2_ds.map(lambda x: (x, 1))

test_ds = temp_ds.concatenate(temp2_ds)

batch_size = 10
test_ds = test_ds.map(load_image_val, num_parallel_calls=16)\
               .batch(batch_size)\
               .repeat()

for images, labels in train_ds.take(1):
    fig, ax = plt.subplots(1, 10, figsize=(20, 6))
    for j in range(10):
        image = images[j].numpy()
        image = image / np.amax(image)
        image = np.clip(image, 0, 1)
        ax[j].imshow(image)
        ax[j].set_title(labels[j].numpy())
plt.show()
```
Model will be saved whenever an improvement in the training is observed at a checkpoint.
```
checkpoint_path = "./train/x-ray/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)
```
The model will then be trained using the data. The length of all the folders are obtained previously.

 ``` 
 history = model.fit(train_ds, 
          steps_per_epoch=train_len/20,
          validation_data=val_ds,
          validation_steps=val_len/20,
          epochs=50,
          verbose=1,
          callbacks=[cp_callback]
          )
```
After it is done, we can obtain the training and validation loss and accuracy.
![GitHub Logo](/images/im2.png)
![GitHub Logo](/images/im3.png)

Clearly the training is excellent on the datasets after 50 epochs since the accuracy has reached almost 98% but the validation has not improved much. The accuracy is ~75% for validation data. This kind of overfitting can be avoided using dropping the data points in the model. Here I will proceed with the model evaluation which shows about 90% accuracy on the test dataset. Of course all these numbers depend on the images and can vary depending on the validation and test datasets.
```
model.evaluate(test_ds, steps=int(test_len/batch_size))
```
We have demonstrated an establishment of image classifier in DNN using Tensorflow. Grey scale chest X-ray images were used for this classification. The initial training dataset can further be extended to include all the images. A local PC platform like mine with a core i-7 8550U took more than 30 minutes to complete the training for 50 epochs on a dataset of 2000 images. A GPU enabled PC will be able to train significantly faster.

