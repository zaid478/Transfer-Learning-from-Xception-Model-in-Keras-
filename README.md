# Transfer-Learning-from-Xception-Model-in-Keras-
This is the Keras code for Transfer Learning.

We will use a pre trained Deep Convolutional Neural Network "Xception" to transfer learn on our own Data.

This model is previously trained on ImageNet Data set , so we need to remove the last fully connected layers and attach some new layers (Fully connected)
based on the number of classes we have. In this case, I am assuming 8 classes.

# Requirements
-> Keras 2.0

-> Numpy

# Important
Read the code line by line
The comments are well organized and written to follow.

You need some changes to make it adaptive to your code. Following the "Comments" will help you out


# Running the Code
python transfer_learn.py
