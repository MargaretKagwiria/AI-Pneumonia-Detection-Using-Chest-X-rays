# AI-Pneumonia-Detection-Using-Chest-X-rays
This is a project focusing on using TensorFlow library to create different CNN(Convolutional Neural Network) to distinguish between normal healthy lungs and sick lungs affected by Pneumonia by using images of Chest X-rays.
A simple CNN and other complex models were created using transfer learning through VGG16 and Mobilenet.
The simple CNN was created using the python script named pneumonia_custom_model.py. 
The python script named mobilenet_variations.py uses transfer learning through mobilenet to create multiple models by training different numbers of layers. Mobilenet was chosen since its a smaller model with a size of approximately 26mb unlike VGG16 which is bulkier with a size of more than 500mb.
