# number-recognisation
A machine learning project called Number Recognition uses deep learning to identify handwritten numbers. The model is useful for tasks like optical character recognition (OCR) and digit classification because it has been taught to recognise digits from 0 to 9. Summary of Contents Overview Installation and Use Results of the Dataset Model Architecture Training Evaluation Introduction to Contributing Licence Number Recognition is a Python-based project that builds and trains a neural network that can recognise handwritten numbers using deep learning tools like TensorFlow and Keras. The model is intended to achieve excellent generalisation and accuracy on new data.

Installation Clone the repository: https://github.com/coderbhawana/number-recognisation.gitdirectory: cd number-recognition Install the required dependencies: pip install -r requirements.txt Usage To run the number recognition model on your own handwritten digit images, follow these steps:

Create PNG files from your digit pictures. Make sure the photographs are grayscale and 28x28 pixels in size. The photographs should be put in the test_images folder. Start the prediction programme by typing python predict.py. The model will analyse the pictures and show the identified numbers along with confidence levels.

Dataset The MNIST dataset, which includes 28x28 grayscale photographs of handwritten digits, was utilised to train the model. 10,000 test photos and 60,000 training images make up the dataset.

The TensorFlow/Keras library contains the MNIST dataset, which the project will automatically download and preprocess while it is being trained.

Architectural Models The deep convolutional neural network (CNN) is the foundation of the number recognition model. Convolutional layers are followed by max-pooling layers, dropout layers to minimise overfitting, and fully linked layers to perform prediction in the architecture.

The model.py file defines the precise model architecture and hyperparameters.

Training Follow these procedures to train the model from scratch or to make adjustments:

Make that the dataset is ready and has been processed. If required, alter the config.py file's values and hyperparameters. Start the training programme by typing python train.py. A record of the training procedure will be kept, and the trained model will be stored for future use.
Evaluation To keep track of the model's accuracy and loss throughout training, its performance is assessed on a different validation set. On a test set not used for training or validation, the model's ultimate performance is assessed after training.

Results Our number recognition model successfully recognises handwritten digits with an accuracy of over 98% on the test set.

Contributing To the project's improvement, we welcome contributions. To contribute, fork the repository, make your modifications in a feature branch, and then submit a pull request.

Thank you for using Number Recognition! If you have any questions or feedback, please don't hesitate to reach out. Happy digit recognition!
