# My-pipeline-for-data-preprocessing-for-image-classification-tasks
Data collection and preprocessing: This includes collecting a large dataset of labeled images, and then preprocessing the images by resizing, normalizing, and augmenting them to ensure that the model is not overfitting.
The functions below are most useful when there are no pandas dataframes provided and you do not want or need to work with dataframes, instead work directly with files and directory classes.

## Table of contents

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#importing-libraries">Importing libraries</a></li>  
    <li><a href="#resetting-a-directory">Resetting a directory</a></li> 
    <li><a href="#getting-some-dataset-stats-overview">Getting some dataset stats overview</a></li> 
    <li><a href="#checking-for-corrupt-image-files">Checking for corrupt image files</a></li>
    <li><a href="#train-test-validation-split">Train Test Validation split</a></li>
    <li><a href="#visualizing-samples">Visualizing samples</a></li>
    <li><a href="#data-augmentation">Data augmentation</a></li>
   
  </ol>
</details>

## Importing libraries
```py
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
import random
from cv2 import imread
import time
```

## Resetting a directory

`reset_directory(dir_path)` removes all files within a directory defined by dir_path.

```py
def reset_directory(dir_path=''):

    # If the directory is already empty, print a message and return

    if len(os.listdir(dir_path)) == 0 :
        print(dir_path + " is already empty")
        return

    # Print a message and record the starting time
    beg = time.time()
    print("resetting "+ dir_path)

    # Delete the directory and all its contents
    shutil.rmtree(dir_path)

    # Create an empty directory in the same location
    os.makedirs(dir_path)
    print(dir_path + " is now empty")
    print("timing : " + str(time.time() - beg))
```

## Getting some dataset stats overview
This function `print_lengths(path = '')` prints the statistics of the images present in the directory and its subdirectories: 
* prints the number of subdirectories present in the directory.
* prints the number of images present in each subdirectories 


```py

def print_lengths(path = ''):
    l =  []
    for class_name in os.listdir(path):
        print("{} : {} images".format(class_name,len(os.listdir(path + '/' + class_name))))
        l.append(len(os.listdir(path + '/' + class_name)))
        
    print("{} CLASSES".format(len(l)))        
    print("TOTAL SIZE = {}".format(sum(l)))    
```

```py

print("*********************")
print("TRAIN DATA STATS")

print_lengths(path = train_path)

print("*********************")
print("TEST DATA STATS")

print_lengths(path = test_path)

print("*********************")
```
This function `print_stats(path = '', verbose = False)` prints statistics of an image dataset by providing the number of images in each subdirectories, `max`,` min`, `sum`, `average` and `standard deviation` of the number of images in each subdirectories. It also returns the list of the number of images in each subdirectories.

```py
def print_stats(path = '', verbose = False):
    """
    Print statistics of an image dataset.

    Args:
        path (str): The directory path of the image dataset.
        verbose (bool): If True, print the number of images in each subdirectory.
    
    Returns:
        list : A list of the number of images in each subdirectory of the dataset.

    """
    sizes = []
    i = 0
    for filename in os.listdir(path):
        i+=1
        size = len(os.listdir(path + '/' + filename))
        sizes.append(size)
        if verbose == True :
            print(size , end = " ")
            if (i% 20 == 0):
                print()
    print()

    print("Max samples = {}".format(max(sizes)))
    print("Min samples = {}".format(min(sizes)))
    print("sum samples = {}".format(sum(sizes)))
    print("Average sample size = {}".format(np.mean(sizes)))
    print("Sample sizes standard deviation = {}".format(np.std(sizes)))

    return sizes
   
```
## Checking for corrupt image files

This function `extract_corrupt_img_files(dir_path='' ,verbose = True)` is used to find the corrupted image files in a given directory and returns the list of corrupted files. It also provides some details like the time taken to process the files, the number of corrupted files found, and their names.

```py
def extract_corrupt_img_files(dir_path='' ,verbose = True):

  i = 0
  beg = time.time()
  corrupted = []
  for filename in os.listdir(dir_path):
    i +=1
    if verbose == True:
        if (i % 50 == 0):
          print(i, end =" ")
        if (i % 1000 == 0):
          print()
    try:
      img = Image.open(dir_path + '/' + filename)
    except:
      corrupted.append(dir_path + '/' + filename)
      continue

  end = time.time()
  print()
  print('*' * 50) 
  print("\nTASK FINISHED IN " + str(end - beg) + " seconds ")
  print("{} corrupted files found in {}".format(len(corrupted), dir_path))
  print(corrupted)
  print()
  print('*' * 50) 
  return corrupted
```
```py
corrupted = []
for path in [test_path, train_path]:
    for class_name in os.listdir(path):
        l = extract_corrupt_img_files(path + '/' + class_name)
        corrupted = corrupted + l

print(len(corrupted))
print(corrupted)
```
## Train Test Validation split

This function `train_test_validation_split(data_path = '',test_split = 0.15, validation_split = 0.15)` is used to split the dataset into three parts: train, test, and validation set. It takes the data path for the dataset and the percentage for each of the test and validation splits.

```py
def train_test_validation_split(data_path = '',test_split = 0.15, validation_split = 0.15):


    # Calculate the total number of files in the dataset
    data_size = len(os.listdir(data_path))

    # Calculate the number of files to include in the test and validation sets
    test_size = int(test_split * data_size)
    validation_size = int(validation_split * data_size)
    

    
    test_sample = []
    validation_sample = []
    train_sample = []
    
    # Select a random sample of files for the test set
    test_sample = random.sample(os.listdir(data_path),test_size )
    
    # Calculate the remaining files that are not in the test set
    train_data = set(os.listdir(data_path)) - set(test_sample)
    train_data = list(train_data)
    
    # Select a random sample of files from the remaining files for the validation set
    validation_sample = random.sample(train_data,validation_size )
    
    # Calculate the remaining files that are not in the validation set
    train_sample = set(train_data) - set(validation_sample)
    train_sample = list(train_sample)
    
    # Print the sizes of the train, test, and validation sets
    print('train size ' + str(len(train_sample)))
    print('test size ' + str(len(test_sample)))
    print('validation size ' + str(len(validation_sample)))
    
    return train_sample, test_sample , validation_sample
```
    

## Visualizing samples

This function `visualize_samples(path ='', ncols = 1, nrows = 1, fig_size = (7,4), title ="")` is used to visualize a random sample of images from a directory. It takes the directory path, number of columns, number of rows, figure size and title as input and display the images in a grid format.

```py

def visualize_samples(path ='', ncols = 1, nrows = 1, fig_size = (7,4), title =""):
    fig = plt.figure(figsize = fig_size)
    
    i = 0
    for filename in (random.sample(os.listdir(path), ncols * nrows)):
        img = imread(path + '/' + filename)
        fig.add_subplot(nrows, ncols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        i+=1
        
    
    plt.tight_layout()
    plt.suptitle(title)
```
```py
for class_name in os.listdir(train_path):
    visualize_samples(train_path + '/' + class_name
                      ,5,2, title = class_name)
```  

## Data augmentation
```py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.9,1.1),
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(rescale=1./255)
```

```py
print("train generator :", end ='')
train_generator = train_datagen.flow_from_directory(
    directory = train_path,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=1,
    subset='training'
)

print("validation generator :", end ='')
validation_generator = train_datagen.flow_from_directory(
    directory = train_path,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=8,
    shuffle=True,
    seed=1,
    subset='validation'
)

print("test generator :" , end ='')
test_generator = test_datagen.flow_from_directory(    
    directory = test_path,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=1
)
```










