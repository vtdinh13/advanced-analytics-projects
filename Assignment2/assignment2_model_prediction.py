import shutil #important for interaction with local machine
import numpy as np
import tensorflow as tf # keras is build on top of tensorflow
from tensorflow import keras
from scipy import ndimage
import os # OperatingSystem, important for same reason as "shutil"
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras import layers
from tensorflow.keras.applications import xception
import pandas as pd

pd.set_option('display.max_rows', 7000)
pd.set_option('display.max_columns', 7000)
pd.set_option('display.width', 7000)

model = keras.models.load_model('/Users/ivoarasin/Desktop/Master/Semester Two/Adv. Analytics in Bus./pythonProjects/Assignment2/modelV2.keras')


# TEST data set to measure real-world performance
input_test_2 = keras.utils.image_dataset_from_directory(
    '/Users/ivoarasin/Desktop/Master/Semester Two/Adv. Analytics in Bus./pythonProjects/Assignment2/Classification Test/random_classif',
    labels=None,
    shuffle=False,
    batch_size=1,
    image_size=(256, 256),
    crop_to_aspect_ratio=True,
    seed=1937465)

# This function retrieves all file_names of the files in the test-dataset
def get_name_list(l):
    name_array = l.file_paths
    for i in range(len(name_array)):
        name_array[i] = name_array[i][130:]
    return name_array

# This block of code evaluates each individual file in the test-set and classifies it by moving it from folder "random_classif" to one of three folders belonging
# to the three labels used in this model, i.e. "Interior" and "Dishes" and "Other". Thus, if you want to adapt this code, make sure to change the paths accordingly
file_names = get_name_list(input_test_2)
input_test_2 = input_test_2.map(lambda x: x / 256)
dir_path = '/Users/ivoarasin/Desktop/Master/Semester Two/Adv. Analytics in Bus./pythonProjects/Assignment2/Classification Test/random_classif/'
class_name_map = ['interior', 'plat', 'other']
set_length = len(input_test_2)
print(file_names)

for i, item in enumerate(input_test_2):
    print(round((i/set_length)*100, 3), "%")
    # pred_ts = tf.nn.sigmoid(model.predict(item))
    pred = model.predict(item)  # pred_ts.numpy()[0][0]
    # print(file_names[i])
    # print(pred[0])
    class_index = np.where(pred[0] == max(pred[0]))[0][0]
    # print('\n')
    class_name = class_name_map[class_index]
    print("name: ", class_name, "; probs.: ", pred)

    #print(class_name)
    def keep_out():
        if class_name == 'interior':
            shutil.move('{}{}'.format(dir_path, file_names[i]),
                        '/Users/ivoarasin/Desktop/Master/Semester Two/Adv. Analytics in Bus./pythonProjects/Assignment2/Classification Test/interior_classif')
        elif class_name == 'plat':
            shutil.move('{}{}'.format(dir_path, file_names[i]),
                        '/Users/ivoarasin/Desktop/Master/Semester Two/Adv. Analytics in Bus./pythonProjects/Assignment2/Classification Test/plat_classif')
        else:
            shutil.move('{}{}'.format(dir_path, file_names[i]),
                        '/Users/ivoarasin/Desktop/Master/Semester Two/Adv. Analytics in Bus./pythonProjects/Assignment2/Classification Test/other_classif')



