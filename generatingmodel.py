import os
import numpy  as np 
import warnings
warnings.filterwarnings("ignore")

import random
from random import seed
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_dir = 'training/'

train_data_generator = ImageDataGenerator(
    #rescale=1./255,
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    horizontal_flip=True
)

validation_data_generator = ImageDataGenerator(
    #rescale=1./255,
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

test_data_generator = ImageDataGenerator(
    #rescale = 1./255,
    preprocessing_function=preprocess_input
)

train_generator = train_data_generator.flow_from_directory(
    dataset_dir,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=128,
    shuffle = True,
    subset="training",
    seed = 43
)

validation_generator = validation_data_generator.flow_from_directory(
    dataset_dir,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=128,
    shuffle = True,
    subset="validation",
    seed = 43
)

test_generator = test_data_generator.flow_from_directory(
    dataset_dir,    
    target_size=(150,150),
    class_mode='categorical',
    batch_size=128,
    shuffle = True
)

# core = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
# core.trainable = False


# model = Sequential([
#     core,
#     Flatten(),
#     Dense(100, activation="relu"),
#     Dropout(0.2),
#     Dense(50, activation="relu"),
#     Dropout(0.2),
#     Dense(1, activation="sigmoid")
# ])

# Define the base model
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
base_model.trainable = False

# Define the input layer
inputs = Input(shape=(150, 150, 3))

# Preprocess the input
x = preprocess_input(inputs)

# Pass through the base model
x = base_model(x, training=False)

# Add custom layers on top of the base model
x = Flatten()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(train_generator.num_classes, activation="sigmoid")(x)

# Define the model
model = Model(inputs, outputs)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])


earlystopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)


checkpoint_path = "model_checkpoint.keras"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)


history = model.fit(train_generator, epochs=20, validation_data=validation_generator, callbacks=[earlystopping, checkpoint])

evaluation = model.evaluate(test_generator)
print(f'test Loss: {evaluation[0]:.4f}')
print(f'test Accuracy: {evaluation[1] * 100:.2f}%')

from tensorflow.keras.preprocessing import image
import numpy as np

def predict_single_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(150, 150))
    
    # Convert the image to an array
    img_array = image.img_to_array(img)
    
    # Expand the dimensions to match the input shape of the model (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image
    img_array = preprocess_input(img_array)
    
    # Predict the class of the image
    prediction = model.predict(img_array)
    
    print(prediction)
    print("*")
    

# Example usage:
img_path = "training/Mushti/image_20.jpg"
prediction = predict_single_image(img_path)
print(f"The predicted class is: {prediction}")