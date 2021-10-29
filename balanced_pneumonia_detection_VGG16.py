import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import imagenet_utils
import itertools
import matplotlib.pyplot as plt

# download the mobilenet model and call the summary function to check out the structure of the model
vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()

#  Process the data
train_path = 'balanced_chest_xray_dataset/train'
valid_path = 'balanced_chest_xray_dataset/valid'
test_path = 'balanced_chest_xray_dataset/test'

data_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
train_batches = data_generator.flow_from_directory(directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = data_generator.flow_from_directory(directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = data_generator.flow_from_directory(directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers [:-20]:
    layer.trainable = False

model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=25,
          verbose=2)

#  Save your fine-tuned model
model.save('models/balanced_chest_xray_vgg16.h5')

test_imgs, test_labels = next(test_batches)
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
def plot_confusion_matrix(cm,classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/ cm.sum(axis=1)[:np.newaxis]
        print('Normalized confusion matrix')

    else:
        print('Confusion matrix without normalization')

    print(cm)
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment='center',
                 color='white' if cm[i,j]>thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['Normal', 'Pneumonia']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


report = classification_report(y_true = test_batches.classes , 
                               y_pred = y_pred , 
                               target_names=['Normal', 'Pneumonia'])
print(report)





