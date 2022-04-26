# -*- coding: utf-8 -*-
"""
20/04/2022 Handwritten Digit Recognition

point of improvement: use different data for testing and validation
"""

import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)


#preprocess the data---------------------------
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)

#num classes = 10 as there are 0 - 9 digits 
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

#make sure it is the same type as the result data 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert it 256 pixels 
x_train /=255
x_test /=255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samp[les')

# create the modeal -------------------------------

batch_size = 128 #how many training samples per run
num_classes = 10 #0-9
epochs = 10 #how many times the training data is used

model = Sequential() #in keras either sequential or functional, sequential go from top down, very organised
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape = input_shape))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #nspatial variance by pooling the max value in a 2x2 grid 
model.add(Dropout(0.25)) # noise cancellation, by randomly drop out values to 0 by 0.25
model.add(Flatten()) # result it one diemnsion for dense later 
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) #softwamax to make sure it adds up to 1 


#loss = los functin, cross entropy used as there is more than one label 

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

#train the model ------------------------

hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test)) #verbose=1, progerss bar
print("The model has been successfully trained")

model.save('mnist.h5')

#evaluate the mode; ----------------------

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss', score[0])
print('Test accuracy', score[1])

