#!/usr/bin/python
 
import numpy as np 
np.random.seed(123) #for reproductibility 

#Keras model module 
from keras.models import Sequential 

#import theano 
from theano import ifelse  

#import keras core layers 
from keras.layers import Dense, Dropout, Activation, Flatten 

#import CNN layers from keras 
from keras.layers import  Convolution2D, MaxPooling2D

#useful Utilities to transform data 
from keras.utils import np_utils

#Load MNIST data 
from keras.datasets import mnist 

#Load pre-shuffled MNIST data into train and test sets 
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print X_train.shape 
#It should show the following
# (6000, 28, 28) : It means we have 60,000 image samples in our training set, and the images are 28 pixels x 28 pixels each. 


#Plotting first sample of X_train 
from matplotlib import pyplot as plt 
plt.imshow(X_train[0]) 

#MNIST images have only one depth in contrast to conventional 3 depth (RGB) 
#Transforming my dataset from having shape (, width, height) to (n, depth, width, height) 

#Reshape input data 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print X_train.shape 

#Convert data type to float32 
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

#Normalize data values to the range [0 1]
X_train /= 255 
X_test /= 255 

#At this point data is ready for modeling 

#Preprocess class labels 
#Convert 1-dimensional class arrays to 10-dimensional class metrices 
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10) 

print Y_train.shape 


###~~~~DEFINE MODEL ARCHITECTURE~~~###

#Declare Sequential model 
model = Sequential()


#Declare CNN input layer 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))

print model.output_shape 

#Add more layers to the model 
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


#Compile model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#Fit Keras model 
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

#Evaluate Keras model 
score = model.evaluate(X_test, Y_test, verbose=0)
