#!/usr/bin/env python3
# -*- encoding: iso-8859-1 -*-

import sys
import rospy
import roslib
import rospkg

import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
device_name = tf.test.gpu_device_name()
print(device_name)

#if device_name != '/device:GPU:0':#
#	raise SystemError('GPU devie not found')
#print('Found GPU at: {}'.format(device_name))


def main(args):
	print("KO")

#def main(args):

#        try:
#                rospy.init_node('n_cnn', anonymous=True)
#                rospack = rospkg.RosPack()

                #Caminho do package
#                global ml_lab1_path
#                ml_lab1_path = rospack.get_path("ml_lab3")
#                ml_lab1_path += "/scripts"

#                print("Done")

#        except KeyboardInterrupt:
#                rospy.loginfo("Shutting down")


if __name__ == "__main__":
        main(sys.argv)
