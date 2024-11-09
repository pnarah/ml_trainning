
# Preprocess the data
```
##Rescla the images

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)
``
# Turn data into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224,224),
                                               batch_size=32,
                                               class_mode='categorical',
                                               )


test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224,224),
                                             batch_size=32,
                                             class_mode='categorical',
                                            )

```
# Convert the ImageDataGenerator to a tf.data.Dataset
```
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # Adjust shape for your data
        tf.TensorSpec(shape=(None,10), dtype=tf.int32)  # Adjust for categorical labels
    )
)

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,10), dtype=tf.int32)
    )
)

# Repeat the datasets for multiple epochs
train_dataset = train_dataset.repeat()
test_dataset = test_dataset.repeat()

# Prefetch to improve performance (optional)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# Calculate steps per epoch
train_steps_per_epoch = len(train_data)  # Equivalent to train_data.samples // batch_size
validation_steps_per_epoch = len(test_data)  # Equivalent to test_data.samples // batch_size
```


## Predict and Plot the images using custom function for multiclass 
Function to predic and Plot the predicted image
```buildoutcfg
def pred_and_plot(model, filename, class_names=class_names):
  #Import the target image and preprocess it 
  image = load_and_prep_image(filename)

  # Do predicton 
  pred = model.predict(tf.expand_dims(image, axis=0))
  if len(pred[0]) > 1:
    pred_class = class_names[tf.argmax(pred[0])]
  else:  
    pred_class = class_names[int(tf.round(pred[0]))]

  # plot the image
  plt.imshow(image)
  plt.title(f" Predicted Class - {pred_class}")
  plt.axis(False)
```
call the prediction and plot function
```buildoutcfg
pred_and_plot(model_6, "03-steak.jpeg")
```

https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/03_convolutional_neural_networks_in_tensorflow.ipynb