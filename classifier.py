from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

train_dir = 'dataset/train/'
val_dir = 'dataset/val/'
train_datagen = ImageDataGenerator(rescale=1.0/ 255,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=False)
val_datagen = ImageDataGenerator(rescale=1.0/ 255)

train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

val_set = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size = 32,
    class_mode = 'binary')

model = define_model()
epoch_steps = len(train_set)
val_steps = len(val_set)
model.fit_generator(
    train_set, 
    steps_per_epoch = epoch_steps,
    epochs = 20,
    validation_data = val_set,
    validation_steps = val_steps)


model.save('final_model.h5')


test_dir = 'dataset/test/'
test_datagen = ImageDataGenerator(rescale=1.0/ 255)
test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size = 8,
    class_mode = 'binary')

model_test = load_model('final_model.h5')

score = model_test.evaluate(test_set)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
