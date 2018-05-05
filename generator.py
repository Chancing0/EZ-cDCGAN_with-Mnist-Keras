from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop,Adam
from keras.utils import np_utils
import keras.backend as K
from keras.models import load_model
import matplotlib.pyplot as plt

import sys

import numpy as np

class generator():
    def __init__(self):
        #load model
        self.model=load_model('generator.h5')
    def save_imgs(self,condition_shape):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r*c, 100))
        gen_imgs = self.model.predict([noise,condition_shape])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("mnist1.png")
        plt.close()

if __name__ == '__main__':
    generator=generator()
    x_label1=[6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]
    x_label2=[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]
    condition_shape1=np_utils.to_categorical(x_label1,num_classes=10)
    condition_shape2=np_utils.to_categorical(x_label2,num_classes=10)
    generator.save_imgs(condition_shape1+condition_shape2)
