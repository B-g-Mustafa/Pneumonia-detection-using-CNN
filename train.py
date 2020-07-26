from __future__ import print_function
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import h5py

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

train_data_dir="./chest_xray/train/"
val_data_dir="./chest_xray/val/"
test_data_dir="./chest_xray/test/"

#DATA Preprocessin
train_datagen=ImageDataGenerator(rescale=1./255)
val_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_data=train_datagen.flow_from_directory(train_data_dir,target_size=(300,300),batch_size=32,class_mode='binary')
val_data=train_datagen.flow_from_directory(val_data_dir,target_size=(300,300),batch_size=32,class_mode='binary')
test_data=train_datagen.flow_from_directory(test_data_dir,target_size=(300,300),batch_size=32,class_mode='binary')

#counting the total no of images in each set

normal_train_img_count=len([i for i in os.listdir(train_data_dir+"NORMAL")])
pneumonia_train_img_count=len([j for j in os.listdir(train_data_dir+"PNEUMONIA")])
print("Normal train data count",normal_train_img_count)
print("Pneumonia train data count",pneumonia_train_img_count)

normal_test_img_count=len([i for i in os.listdir(test_data_dir+"NORMAL")])
pneumonia_test_img_count=len([j for j in os.listdir(test_data_dir+"PNEUMONIA")])
print("NOrmal data count",normal_test_img_count)
print("Pneumonia data count",pneumonia_test_img_count)

#since normal class data is very less we shols assign class weight to it
class_zero=((normal_train_img_count+pneumonia_train_img_count)/normal_train_img_count)/2
class_one=((normal_train_img_count+pneumonia_train_img_count)/pneumonia_train_img_count)/2
class_weight={0:class_zero,1:class_one}
print("weight of class 0:{} \nweight of class 1:{}".format(class_weight[0],class_weight[1]))

#Creating Model
model = Sequential([

        #Adding 5 Convolution Layers
        Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
        MaxPooling2D(2,2),

        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        #Flattenning the Pooled Parameters
        Flatten(),

        #Creating Fully Connected Networks of 512 Neurons followed by 128 Neurons
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),


        #Final Prediction is Binary: NORMAL or PNEUMONIA, so using single output neuron with sigmoid funtion
        #to give output in 0-1 where 0 for NORMAL and 1 for PNEUMONIA
        Dense(1, activation='sigmoid')
    ])

model.summary()

#compiling model

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy',keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall')])

#training model

history = model.fit(train_data,steps_per_epoch = 32,epochs = 20,validation_data =test_data ,class_weight = class_weight)


model.save("./pred01.model")
