import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

train_path = 'balanced_chest_xray_dataset/train'
valid_path = 'balanced_chest_xray_dataset/valid'
test_path = 'balanced_chest_xray_dataset/test'

data_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
train_batches = data_generator.flow_from_directory(directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = data_generator.flow_from_directory(directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = data_generator.flow_from_directory(directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)


simple_cnn = Sequential([
    Conv2D(16, kernel_size=(3,3), input_shape=(224,224,3), activation='relu', padding='same'),
    Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
    Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    Flatten(),
    Dense(2, activation='softmax')
])

simple_cnn.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

simple_cnn.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=25,
          verbose=2)

#  Save your fine-tuned model
simple_cnn.save('models/simple_cnn_balanced_dataset.h5')


