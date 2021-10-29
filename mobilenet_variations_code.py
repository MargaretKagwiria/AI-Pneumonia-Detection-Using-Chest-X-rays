import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import random
import shutil
import matplotlib.pyplot as plt


mobile = tf.keras.applications.mobilenet.MobileNet()
mobile.summary()

train_path = r'/content/drive/MyDrive/Machine_Learning_Projects/equal_chest_xray_dataset/train'
valid_path = r'/content/drive/MyDrive/Machine_Learning_Projects/equal_chest_xray_dataset/valid'
test_path = r'/content/drive/MyDrive/Machine_Learning_Projects/equal_chest_xray_dataset/test'
data_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

train_batches = data_generator.flow_from_directory(directory=train_path,
                                                   target_size=(224,224),
                                                   batch_size=10)

valid_batches = data_generator.flow_from_directory(directory=valid_path,
                                                   target_size=(224,224),
                                                   batch_size=10)

test_batches = data_generator.flow_from_directory(directory=test_path,
                                                  target_size=(224,224),
                                                  batch_size=10,
                                                  shuffle=False)

#layers with parameters starting from the output layer
train_layers = []
for l in range(5,87,3):
    train_layers.append(l)

print(train_layers)
#parameter_layers

for n in train_layers:
    mobile = tf.keras.applications.mobilenet.MobileNet()
    #  Build the fine-tuned model
    x = mobile.layers[-6].output
    output = Dense(units=2, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=output)

    for layer in model.layers[:-n]:
        layer.trainable = False

    print(f"training model {n}")
    #  Compile and train the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x=train_batches,
                        steps_per_epoch=len(train_batches),
                        validation_data=valid_batches,
                        validation_steps=len(valid_batches),
                        epochs=25,
                        verbose=2)

    #  Save your fine-tuned model
    model.save(f'/content/drive/MyDrive/Machine_Learning_Projects/models/mobilenet-full-{n}.h5')
    print(f"model {n} saved successfully")