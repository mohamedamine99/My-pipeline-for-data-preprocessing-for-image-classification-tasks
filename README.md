# My-pipeline-for-data-preprocessing-for-image-classification-tasks
Data collection and preprocessing: This includes collecting a large dataset of labeled images, and then preprocessing the images by resizing, normalizing, and augmenting them to ensure that the model is not overfitting.

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

```py

def print_stats(path = ''):
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

print_stats(path = train_path)

print("*********************")
print("TEST DATA STATS")

print_stats(path = test_path)

print("*********************")
```
## Checking for corrupt image files

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










