import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.io import wavfile
import numpy as np
import cv2
import pandas as pd

def_path_audio_val = '/home/rick/Documents/image_in_audio_steganography/all_stft_val/'
def_path_images = '/home/morty/Documents/image_in_audio_steganography/data/VOC2012/JPEGImages/'
def_audio_path_train = '/home/morty/Documents/image_in_audio_steganography/libri_stft/'


def stu(audio_file,def_audio_path):
    sample_rate = 16000
    audio_file_1 = audio_file
    samples = np.load(audio_file_1)
    return samples


def pre_process_img(image_path):
    image_path = image_path
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (513,512))
    return img

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,imgs, batch_size=32, dim_img=(512,513,1), dim_audio=(512,513,1), shuffle=True,def_audio_path= def_audio_path_train):
        'Initialization'
        # self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.dim_img = dim_img
        self.imgs = imgs
        self.dim_audio = dim_audio
        self.shuffle = shuffle
        self.def_audio_path=def_audio_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indexes_imgs = self.indexes_imgs[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_IDs_temp_imgs = [self.imgs[k] for k in indexes_imgs]

       # Generate data
        audio, images = self.__data_generation(list_IDs_temp, list_IDs_temp_imgs)


        return [images/255.0,audio], [np.abs(audio), images/255.0] 

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        self.indexes_imgs = np.arange(len(self.imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.indexes_imgs)

    def __data_generation(self, list_IDs_temp, list_IDs_temp_imgs):
        'Generates data containing batch_size samples' 
        # Initialization
        images = np.empty((self.batch_size, *self.dim_img))
        audio  = np.empty((self.batch_size, *self.dim_audio), dtype=np.complex64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            audio[i, ] =  np.reshape(stu(ID,self.def_audio_path),(512,513,1))
        for j ,ID_img in enumerate(list_IDs_temp_imgs):
            images[i,] = np.reshape(pre_process_img(ID_img), (512, 513, 1))

        return audio,images 