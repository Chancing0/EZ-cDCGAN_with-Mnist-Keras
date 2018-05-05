from __future__ import print_function, division
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
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
    condition=[1,2,3,4,5,6,7,8,9,5,4,3,2,1,0,7]
    image=np_utils.to_categorical(condition,num_classes=10)
    generator.save_imgs(image)
