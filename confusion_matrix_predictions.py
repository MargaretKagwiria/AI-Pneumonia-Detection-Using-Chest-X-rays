import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

test_path = r'/content/drive/MyDrive/Machine_Learning_Projects/equal_chest_xray_dataset/test'

data_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
test_batches = data_generator.flow_from_directory(directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)
model_folder = os.listdir('/content/drive/MyDrive/Machine_Learning_Projects/models')

for model in model_folder:
    fine_tuned_model = load_model(f'/content/drive/MyDrive/Machine_Learning_Projects/models/{model}')

    #  Predict with the model using a confusion matrix
    test_imgs, test_labels = next(test_batches)
    predictions = fine_tuned_model.predict(x=test_batches, steps=len(test_batches), verbose=0)

    cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    def plot_confusion_matrix(cm,classes,
                              normalize=False,
                              title=f'{model}',
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
            print(f'{model}')

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

    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title=f'{model}')

    report = classification_report(y_true = test_batches.classes ,
                                   y_pred = np.argmax(predictions, axis=-1) ,
                                   target_names=['Normal', 'Pneumonia'])
    print(report)

