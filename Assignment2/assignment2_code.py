import shutil #important for interaction with local machine
import numpy as np
import tensorflow as tf # keras is build on top of tensorflow
from tensorflow import keras
from matplotlib import pyplot as plt
import os # OperatingSystem, important for same reason as "shutil"

# Base Model
base_model = keras.applications.Xception( #Xception is a pre-trained model trained on more than 1m images and is able to classify into 1000 categories
    weights='imagenet', # Pre-trained weights from ImageNet
    input_shape=(256,256,3),
    include_top=False) # Discards top layer of ImageNet Classifier

# Freeze model, so that weights remain the same even after training on new data
base_model.trainable = False

# Explanation of "Input Shape"
# if you have 30 images of 256x256 pixels in RGB (3 channels), the shape of your input data is (30,256,256,3)
input_shape = keras.Input(shape=(256,256,3)) # define shape of input for model

x = base_model(input_shape, training=False) 
x = keras.layers.GlobalAvgPool2D()(x) # a flatten operation. But not fully sure what it does, have to study it further
output = keras.layers.Dense(1)(x) # New output layer that is trainable so that pre-trained model can actually capture and recognize on new classes
model = keras.Model(input_shape, output) # assembling model

# Retrieving train data which is separated into two folders in my case where folder name (i.e. directory structure) reflects class-label of the data
# E.g. I have two folders, one called "Interior" and one named "Dishes" in the parent-folder of "Assignment2DataInput"
input_train = keras.utils.image_dataset_from_directory('/Users/ivoarasin/Desktop/Master/Semester Two/Adv. Analytics in Bus./pythonProjects/Assignment2/Assignment2DataInput',
                                         labels='inferred', # important so that keras takes folder names as class-labels
                                         label_mode='binary', # will convert string names into integer labels
                                         batch_size=16, # for smaller less powerful machines, smaller batch sizes are recommendable
                                         image_size=(256, 256), # image size definition
                                         validation_split=0.2, # obvious
                                         subset='training', # 80% train
                                         crop_to_aspect_ratio=True, # since the neural net can only take imput if it is in standard size, use this argument
                                         seed=1937465)

input_valid = keras.utils.image_dataset_from_directory('/Users/ivoarasin/Desktop/Master/Semester Two/Adv. Analytics in Bus./pythonProjects/Assignment2/Assignment2DataInput',
                                         labels='inferred',
                                         label_mode='binary',
                                         batch_size=16,
                                         image_size=(256, 256),
                                         validation_split=0.2,
                                         subset='validation', # 20% for model tuning
                                         crop_to_aspect_ratio=True,
                                         seed=1937465)

val_batches = tf.data.experimental.cardinality(input_valid)
#input_test = input_valid.take((2*val_batches)//3)  # TEST SET
input_valid = input_valid.skip((2*val_batches)//3) # VALIDATION SET FOR TUNING

input_train = input_train.map(lambda x,y: (x/256, y)) # model requires image domain to be within [-1, 1], so normalize all image pixel matrices
input_valid = input_valid.map(lambda x,y: (x/256, y))
#input_test = input_test.map(lambda x,y: (x/256, y))

plt.figure(figsize=(10, 10)) # if you want to visualize a few images along with their labels, this is the way to do it#
# Careful though, often times below code must be slightly adapted for individual purposes, don't expect it to work right out of the box
for i, (image, label) in enumerate(input_train.take(16)):
    ax = plt.subplot(6, 6, i + 1)
    plt.imshow(image[i])#.numpy().astype("uint8"))
    plt.title(int(label[i]))
    plt.axis("off")
#plt.show()

data_augmentation = keras.Sequential( # Data Augmentation function. Not used here since model performs well enoguh, but kept it in for possible later needs
    [tf.keras.layers.RandomFlip('horizontal'),
     tf.keras.layers.RandomRotation(0.1),]
)

model.compile(optimizer=keras.optimizers.Adam(), # Adam is an optimizer developed by OpenAIs CEO Ilya Sutskever, what a coincidence haha, can you imagine?
              loss=keras.losses.BinaryCrossentropy(from_logits=True), # define loss function, not sure what it does exactly, have to study it further
              metrics=[keras.metrics.BinaryAccuracy()]) 
print(model.summary())


model.fit(input_train, epochs=2, validation_data=input_valid) # fitting model

# TEST data set to measure real-world performance
input_test_2 = keras.utils.image_dataset_from_directory('/Users/ivoarasin/Desktop/Classification Test/random_classif',
                                         labels=None,
                                         shuffle=False,
                                         batch_size=1,
                                         image_size=(256, 256),
                                         crop_to_aspect_ratio=True,
                                         seed=1937465)


# some output you could visualize
#loss, acc = model.evaluate(input_test_2)
#preds = tf.nn.sigmoid(model.predict(input_test_2))
#print('Test accuracy: ', acc)
#print('predictions: ', preds)


# This function retrieves all file_names of the files in the test-dataset
def get_name_list(l):
    name_array = l.file_paths
    for i in range(len(name_array)):
        name_array[i] = name_array[i][60:]
    return name_array

# This block of code evaluates each individual file in the test-set and classifies it by moving it from folder "random_classif" to one of two folders belonging
# to the two labels used in this model, i.e. "Interior" and "Dishes". Thus, if you want to adapt this code, make sure to change the paths accordingly
file_names = get_name_list(input_test_2)
input_test_2 = input_test_2.map(lambda x: x/256)
dir_path = '/Users/ivoarasin/Desktop/Classification Test/random_classif/'
for i, item in enumerate(input_test_2):
    pred_ts = tf.nn.sigmoid(model.predict(item))
    pred = pred_ts.numpy()[0][0]
    #print(file_names[i])
    #print(pred)
    #print('\n')
    if pred <= 0.5: # moves image from unknown label to "Interior"
        shutil.move('{}{}'.format(dir_path,file_names[i]), '/Users/ivoarasin/Desktop/Classification Test/interior_classif')
    else:           # moves image from unknown label to "Dishes" (called "plat_classif" in my case) 
        shutil.move('{}{}'.format(dir_path,file_names[i]), '/Users/ivoarasin/Desktop/Classification Test/plat_classif')



#plt.figure(figsize=(10, 10))
#for i, image in enumerate(input_test_2):
#     ax = plt.subplot(6, 6, i + 1)
#     plt.imshow(image[0])#.numpy().astype("uint8"))
#     plt.title(file_names[i])
#     plt.axis("off")
#     print(image)
#plt.show()
