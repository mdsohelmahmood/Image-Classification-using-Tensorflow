# X-ray Image Classification and Model Evaluation
## Pneumonia detection from chest X-ray images using Tensorflow

![GitHub Logo](/images/owen-beard-DK8jXx1B-1c-unsplash.jpg)
Format: ![Alt Text](url)

Kaggle has a wonderful source of chest X-ray image datasets for pneumonia and normal cases. There are significant differences between the image of a normal X-ray and an affected X-ray. Machine learning can play a pivotal role in determining the disease and significantly boost the diagnosis time as well as reduce human effort. In this article, I will walk through this dataset and classify the images with an evaluation accuracy of 90%.

![GitHub Logo](/images/im1.png)
Format: ![Alt Text](url)

I have been motivated by the work done here on the datasets between cats and dogs and reused the code block for dataset pipeline. First we need to import the necessary packages.

The initial dataset is too big. I have chosen a reduced set having 1000 images for normal case and 1000 images for pneumonia. There will be three directories: train, validation and test. Validation and test datasets will be completely new to the model and we don’t need to perform any image augmentation technique to these datasets.

Next few samples are imported. The output shows the X-ray images from the targeted two cases.

We will assign labels to the datasets: 0 for NORMAL and 1 for PNEUMONIA. That’s why we will use SparseCategoricalCrossentropy() loss function in the model compilation. The model is defined as below:

Since there are equal number of images in each of the target folders, the total train or validation or test length is obtained by multiplying one folders length by 2. Next part is the image augmentation

We will flow training images in batches of 20 using train_datagen and validation_datagen generator.

A bunch of functions to manipulate the images are placed.

These functions will induce some random jittering and help to load images. The next block will assign the labels to the individual folders. As mentioned earlier, 0 will be assigned for NORMAL and 1 will be assigned for PNEUMONIA. This is applied for all the three folders of train, validation and test.

Model will be saved whenever an improvement in the training is observed at a checkpoint.

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

Clearly the training is excellent on the datasets after 50 epochs since the accuracy has reached almost 98% but the validation has not improved much. The accuracy is ~75% for validation data. This kind of overfitting can be avoided using dropping the data points in the model. Here I will proceed with the model evaluation which shows about 90% accuracy on the test dataset. Of course all these numbers depend on the images and can vary depending on the validation and test datasets.

We have demonstrated an establishment of image classifier in DNN using Tensorflow. Grey scale chest X-ray images were used for this classification. The initial training dataset can further be extended to include all the images. A local PC platform like mine with a core i-7 8550U took more than 30 minutes to complete the training for 50 epochs on a dataset of 2000 images. A GPU enabled PC will be able to train significantly faster.

