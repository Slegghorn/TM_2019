import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random

#data variables
DATA_dir = 'C:\\Users\\yarne\\dropbox\\deeplearning\\Data\\Petimages'
categories = ['Dog', 'Cat']
img_size = 50

#create train_data
train_data = []
def create_train_data():
    for category in categories:
        path = os.path.join(DATA_dir, category)
        class_num = [0, 0]
        class_num[categories.index(category)] = 1
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (img_size, img_size))
                train_data.append([img_array, class_num])
            except Exception as e:
                pass
create_train_data()
random.shuffle(train_data)

x = []
y = []

for features, label in train_data:
    x.append(features)
    y.append(label)
x = np.array(x).reshape(-1, img_size, img_size, 1)
np.save('x_data', x)
np.save('y_data',y)
