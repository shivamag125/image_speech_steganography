import tensorflow as tf
from tensorflow import keras
import numpy as np
import numpy as np
import keras
import cv2
import pandas as pd

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, num_imgs, batch_size=8, dim_img=(256,256,3), shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.dim_img = dim_img
        self.num_imgs = num_imgs
        self.indexes = np.arange(0, num_imgs)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_imgs / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

       # Generate data
        x, y = self.__data_generation(indexes)
        return x, y 

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0, self.num_imgs)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        # Initialization
        images = np.empty((self.batch_size, *self.dim_img))
        y = np.empty((self.batch_size, *self.dim_img))

        for i in range(self.batch_size):
            img1 = cv2.imread('/home/rick/Documents/image_in_audio_steganography/Steg/scripts_V5/Good_Dataset/clean/'+str(indexes[i])+'.png')
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread('/home/rick/Documents/image_in_audio_steganography/Steg/scripts_V5/Good_Dataset/dirty/'+str(indexes[i])+'.png')
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            images[i] = cv2.resize(img2, (256, 256))/255
            y[i] = cv2.resize(img1, (256,256))/255

        return images, y 