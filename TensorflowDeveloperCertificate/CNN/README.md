# 1. Get ZIP data set and extract 
```buildoutcfg
import zipfile

!wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip

# Unzip the extracted file
zip_ref = zipfile.ZipFile("pizza_steak.zip")
zip_ref.extractall()
zip_ref.close()
```

# 2. OS walk-through a dir and count files
```buildoutcfg
import os
# walk through a directory and list number of files

for dirpath, dirnames, filenames in os.walk('pizza_steak'):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")


# anotjer way to find numver of files in a directory

num_steak_images_train = len(os.listdir('pizza_steak/train/steak'))
num_steak_images_train
```

# 3. View image shape 
```
img.shape
## Returns Width, Height , Colour channels
(512, 512, 3)
```

# 4. Load data from directory and turn them into batched
## 4.1 Define data directories
```buildoutcfg
train_dir = "pizza_steak/train/"
test_dir = "pizza_steak/test/"
```
## 4.2 Create train and test data generator and rescale
```buildoutcfg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='binary'
                                               )

test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224,224),
                                             batch_size=32,
                                             class_mode='binary'
                                             )
```


# 5. Get classname programatically from the directory 
```
import pathlib
import numpy as np
data_dir = pathlib.Path('pizza_steak/train')
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
print(class_names)
```
# 6. View random Image 

## 6.1 Function to visualize random images 

```buildoutcfg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import random

def view_random_image(target_dir, target_class):
    # setup target directory
    target_folder = target_dir + target_class

    #Get one random image from target folder
    random_image = random.sample(os.listdir(target_folder), 1)

    #Read image and plor using matplot 
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")

    print(f"Image shape {img.shape}") # show shape of the image 
    return img
```


## 6.2 View random image
```buildoutcfg
img = view_random_image(target_dir="pizza_steak/train/", target_class='steak')
```


#7. Predict a custom Image 

## Load and resize the image to the model image shape
Create the function to load and reshape custom image
```buildoutcfg
def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, color_channels)
  """
  # Read image and conver to tensor
  img = tf.io.read_file(filename)
  img = tf.image.decode_image(img)
  # Resize the image
  img = tf.image.resize(img, size=[img_shape, img_shape])

  # Rescale the image
  img = img/255. 
  return img

```
## Call the load_and_prep_image function and do prediction
```buildoutcfg
steak = load_and_prep_image("03-steak.jpeg")
steak.shape

# Get the class names (programmatically, this is much more helpful with a longer list of classes)
import pathlib
import numpy as np
data_dir = pathlib.Path("pizza_steak/train/") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)

# Predict 
pred = model_6.predict(tf.expand_dims(steak, axis=0))

# Get the predicted calss
pred_class = class_names[int(tf.round(pred))]
pred_class
```
## Predict and Plot the images using custom function
Function to predic and Plot the predicted image, this will work with binary classification only
```buildoutcfg
def pred_and_plot(model, filename, class_names=class_names):
  #Import the target image and preprocess it 
  image = load_and_prep_image(filename)

  # Do predicton 
  pred = model.predict(tf.expand_dims(image, axis=0))
  pred_class = class_names[int(tf.round(pred))]

  # plot the image
  plt.imshow(image)
  plt.title(f" Predicted Class - {pred_class}")
  plt.axis(False)
```
call the prediction and plot function
```buildoutcfg
pred_and_plot(model_6, "03-steak.jpeg")
```