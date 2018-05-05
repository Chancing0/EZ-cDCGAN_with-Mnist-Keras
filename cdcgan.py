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

import matplotlib.pyplot as plt

import sys

import numpy as np

class cdcgan():
    def __init__(self):
        # Image size 
        self.img_rows=28
        self.img_cols=28
        self.channels=1
        self.condition_dim=10
        # Optimizer
        self.optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator=self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=self.optimizer,metrics=['accuracy'])

        # Build and compile the generator
        self.generator=self.build_generator()
        #self.generator.compile(loss='binary_crossentropy',optimizer=self.optimizer,metrics=['accuracy'])

        # Input
        z=Input(shape=(100,))
        condition_shape=Input(shape=(self.condition_dim,))
        img=self.generator([z,condition_shape])
        self.discriminator.trainable=False

        vaild=self.discriminator([img,condition_shape])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined=Model(inputs=[z,condition_shape],outputs=vaild)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer,metrics=['accuracy'])

    def build_generator(self):

        noise_shape = Input(shape=(100,))
        condition_shape=Input(shape=(self.condition_dim,))
        #model = Sequential()
        generator_input = concatenate([noise_shape, condition_shape])
        x=Dense(128 * 7 * 7, activation="relu")(generator_input)
        x=Reshape((7, 7, 128))(x)
        x=BatchNormalization(momentum=0.8)(x)
        x=UpSampling2D()(x)
        x=Conv2D(128, kernel_size=4, padding="same")(x)
        x=Activation("relu")(x)
        x=BatchNormalization(momentum=0.8)(x)
        x=UpSampling2D()(x)
        x=Conv2D(64, kernel_size=4, padding="same")(x)
        x=Activation("relu")(x)
        x=BatchNormalization(momentum=0.8)(x)
        x=Conv2D(1, kernel_size=4, padding="same")(x)
        x=Activation("tanh")(x)

        #model.summary()

        #define input layer
        #noise = Input(shape=noise_shape)
        #img = model(noise)

        return Model(inputs=[noise_shape, condition_shape],outputs=x)

    def build_discriminator(self):

        img_shape = Input(shape=(self.img_rows, self.img_cols, self.channels))
        condition_shape=Input(shape=(self.condition_dim,))
        di1=Reshape((1, 1, self.condition_dim))(condition_shape)
        di1=UpSampling2D((self.img_rows, self.img_cols))(di1)
        discriminator_input = concatenate([img_shape, di1])

        x=Conv2D(16, kernel_size=3, strides=2,padding="same")(discriminator_input)
        x=LeakyReLU(alpha=0.2)(x)
        x=Dropout(0.25)(x)
        x=Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
        x=ZeroPadding2D(padding=((0,1),(0,1)))(x)
        x=LeakyReLU(alpha=0.2)(x)
        x=Dropout(0.25)(x)
        x=BatchNormalization(momentum=0.8)(x)
        x=Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x=LeakyReLU(alpha=0.2)(x)
        x=Dropout(0.25)(x)
        x=BatchNormalization(momentum=0.8)(x)
        x=Conv2D(128, kernel_size=3, strides=1, padding="same")(x)
        x=LeakyReLU(alpha=0.2)(x)
        x=Dropout(0.25)(x)

        x=Flatten()(x)
        x = Dense(1, activation="sigmoid")(x)
        #model.summary()

        #define input layer
        #img = Input(shape=img_shape)
        #features = model(img)
        #valid = Dense(1, activation="sigmoid")(features)

        return Model(inputs=([img_shape,condition_shape]),outputs=(x))

    def train(self,epochs,batch_size=128, save_interval=50):
        # Load the dataset
        (x_train,x_label),(_,_)=mnist.load_data()

        # Rescale -1 to 1
        x_train=(x_train.astype(np.float32)-127.5)/127.5
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_label=np_utils.to_categorical(x_label,num_classes=10)
        # batch size
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx=np.random.randint(0,x_train.shape[0],half_batch)
            imgs=x_train[idx]
            condition_shape=x_label[idx]
            noise=np.random.normal(0,1,(half_batch,100))
            fake=self.generator.predict([noise,condition_shape])

            # Train the discriminator
            d_loss_real=self.discriminator.train_on_batch([imgs,condition_shape],np.ones((half_batch,1)))
            d_loss_fake=self.discriminator.train_on_batch([fake,condition_shape],np.zeros((half_batch,1)))
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise=np.random.normal(0,1,(half_batch,100))

            g_loss=self.combined.train_on_batch([noise,condition_shape],np.ones((half_batch,1)))

             # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch,condition_shape)
                self.generator.save("generator.h5")
            
    def save_imgs(self, epoch,condition_shape):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r*c, 100))
        gen_imgs = self.generator.predict([noise,condition_shape])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images\\mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cdcgan = cdcgan()
    cdcgan.train(epochs=90000, batch_size=32, save_interval=50)




