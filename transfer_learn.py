from keras.models import Model
from keras.layers import Dense,Flatten,GlobalAveragePooling2D

from keras.applications import xception
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import ModelCheckPoint
import glob
import numpy as np
import scipy as scp

# X should contain your training Data/Images
# Y should be your Labels

# set the number of classes according to the data
num_classes=8

# set according to your choice of train/test distrbutions
num_training_samples=3200
num_channels=3
width=720
height=576

# shuflling the data set
permute=np.random.permutation(len(X))
X=X[p]
Y=Y[p]



X_train=X[:num_training_samples,:]
Y_train=Y[:num_training_samples]

X_test=X[num_training_samples:,:]
Y_test=Y[num_training_samples:]

print 'Shape of training Data'
print np.shape(X_train)
print np.shape(Y_train)
print np.shape(X_test)
print np.shape(Y_test)

y_train = np_utils.to_categorical(Y_train, num_classes)
y_test = np_utils.to_categorical(Y_test, num_classes)

# Transfer Learning!!
 
# Importing Xception pre trained model on ImageNet
# include_top=False indicates the fact that we are not importing last fully connected layers of the network

# Tensorflow requires channels in the last for input shape
model = xception.Xception(weights='imagenet', include_top=False, input_shape=(height, width, num_channels))
print (model.summary())

# New Layers which will be trained on our data set and will be stacked over the Xception Model
x=model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dens(512,activation='relu')(x)
output=Dense(num_classes,activation='softmax')(x)


print 'Stacking New Layers'

New_Model=Model(model.input,output)


# Freezing all the Imported Layers
for layers in model.layers:
	layers.trainable=False





import os.path

# If model stop due to some reason , it will continue from the same epoch afterwards
# we are saving the weights after each eopch

if os.path.exists("weight_mediaeval.h5"):
        print ("loading ", "weight_mediaeval.h5")
        model.load_weights('weight_mediaeval.h5')

checkpoint = ModelCheckpoint('weight_mediaeval.h5',verbose=1);


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


print (New_Model.summary())

print 'Model Training'

New_Model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

New_Model.fit(X_train,y_train,batch_size=64,epochs=500,validation_data=(X_test,y_test),shuffle=True,callbacks=[checkpoint])










