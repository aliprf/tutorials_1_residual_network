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

from network_model import NetworkModel


class Train:
    def __init__(self, input_shape, output_shape):
        """

        :param input_shape: shape of input images, default is 28 * 28 *1
        :param output_shape: number of classes. for mnist_ds, default is 10

        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def train_model(self):
        """"""

        '''Loading data and creating dataset:'''
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # x_train shape is 60000*28*28  . y_train is 6000*1
        # x_test shape is 60000*28*28   . y_test is 10000*1

        '''take 5,000 samples from train to be used as validation set'''
        x_val = x_train[-5000:]
        y_val = y_train[-5000:]
        x_train = x_train[:-5000]
        y_train = y_train[:-5000]

        '''adopting the images shape, from ?*28*28 to ?*28*28*1'''
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        '''normalizing images'''
        x_train = x_train/255.0
        x_val = x_val/255.0
        x_test = x_test/255.0

        '''create categorical labels'''
        y_train = keras.utils.to_categorical(y_train, self.output_shape)
        y_val = keras.utils.to_categorical(y_val, self.output_shape)
        y_test = keras.utils.to_categorical(y_test, self.output_shape)

        '''creating model:'''
        net_model = NetworkModel()
        model_1 = net_model.sample_res_net_v0(input_shape=self.input_shape, output_shape=self.output_shape)

        '''preparing network for being trained'''
        model_1.compile(loss=losses.categorical_crossentropy,
                        optimizer=adam(),
                        metrics=['accuracy'])

        '''start training the model'''
        history = model_1.fit(x_train, y_train,
                              batch_size=100,
                              epochs=2,
                              validation_data=(x_val, y_val))
        model_1.save("res_model.hd5")

        self.show_figures(history.history)


    def show_figures(self, history):
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'])
        plt.savefig('accuracy')

        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'])
        plt.savefig('loss')