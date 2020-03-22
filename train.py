import tensorflow as tf
import keras

tf.logging.set_verbosity(tf.logging.ERROR)
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from datetime import datetime
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import os.path
from keras import losses
from keras import backend as K

from model import NetworkModel


class Train:
    def __init__(self, input_shape, output_shape):
        """

        :param input_shape: shape of input images, default is 224 * 224 *3
        :param output_shape: number of classes. for mnist_ds, default is 10

        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def train_model(self):
        net_model = NetworkModel()

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # x_train shape is 60000*28*28  . y_train is 6000*1
        # x_test shape is 60000*28*28   . y_test is 10000*1

        model_1 = net_model.sample_res_net_v0(input_shape= self.input_shape, output_shape= self.output_shape)

        '''preparing network for being trained'''
        model_1.compile(loss=losses.categorical_crossentropy,
                        optimizer=adam(lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False),
                        metrics=['accuracy'])

        history = model_1.fit()
