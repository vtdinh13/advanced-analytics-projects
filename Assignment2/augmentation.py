import shutil #important for interaction with local machine
import numpy as np
#import tensorflow as tf # keras is build on top of tensorflow
#from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image
import os # OperatingSystem, important for same reason as "shutil"

# Code to augment data of label "other" and save it to disk
other_path = '/Users/ivoarasin/Desktop/Master/Semester Two/Adv. Analytics in Bus./pythonProjects/Assignment2/Assignment2DataInput/other'
other_imgs = keras.utils.image_dataset_from_directory(other_path,
                                                      labels=None,
                                                      shuffle=False,
                                                      batch_size=1,
                                                      image_size=(512, 512)
                                                      )

brightness = tf.keras.layers.RandomBrightness([-0.45, 0.45])
contrast = tf.keras.layers.RandomContrast(0.25)
rotation = tf.keras.layers.RandomRotation(0.1)

def img_augmentation(img):
    return brightness(contrast(rotation(img)))

def get_name_list(l, source_path):
    path_len = len(source_path)
    name_array = l.file_paths
    for i in range(len(name_array)):
        name_array[i] = name_array[i][int(path_len):]
    return name_array

def augment_and_write_to_disk(img_set, path, reps_per_image):
    names = get_name_list(img_set, path)
    for idx, img in enumerate(img_set):
        for i in range(reps_per_image):
            aug_img = img_augmentation(img)
            #print(aug_img[0])
            aug_img_array = keras.preprocessing.image.img_to_array(aug_img[0])
            #print(aug_img_array)
            print('{}{}_{}{}'.format(path, names[idx][:-4], i, '.jpg'))
            keras.preprocessing.image.save_img('{}{}_{}{}'.format(path, names[idx][:-4], i, '.jpg'), aug_img_array)
            #aug_img.save('{}{}_{}'.format(path, names[idx], i))
            #shutil.move('{}{}'.format(path, names[idx]), '{}{}_{}'.format(path, names[idx], i))

augment_and_write_to_disk(other_imgs, other_path, 3)

