import numpy as np
import tensorflow as tf
from tensorflow import keras
import steg_model_v3
import decoder_model
from data_loader_imgs_saved_stft_v3 import DataGenerator
import glob
import cv2
import pandas as pd
from tb import tensorboard_custom
import os

def denoiser_model(img_shape=(256,256,3)):
	input_img = keras.layers.Input(shape = img_shape)
	conv1 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_img)
	conv2 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv1)

	pool1 = keras.layers.MaxPooling2D()(conv2)
	conv3 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(pool1)
	conv4 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(conv3)

	pool2 = keras.layers.MaxPooling2D()(conv4)
	conv5 = keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(pool2)

	upsample1 = keras.layers.UpSampling2D()(conv5)
	conv6 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(upsample1)
	conv7 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(conv6)

	c7_merge = keras.layers.Conv2D(128, (1,1), padding='same')(conv7)
	c4_merge = keras.layers.Conv2D(128, (1,1), padding='same')(conv4)
	merge1 =keras.layers.Add()([c7_merge, c4_merge])
	act1 = keras.layers.Activation('relu')(merge1)

	upsample2 = keras.layers.UpSampling2D()(act1)
	conv8 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(upsample2)
	conv9 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv8)

	c2_merge = keras.layers.Conv2D(64, (1,1), padding='same')(conv2)
	c9_merge = keras.layers.Conv2D(64, (1,1), padding='same')(conv9)
	merge2 = keras.layers.Add()([c2_merge, c9_merge])
	act2 = keras.layers.Activation('relu')(merge2)

	output = keras.layers.Conv2D(3, (3,3), padding='same', activation='sigmoid')(act2)

	model = tf.keras.models.Model(input_img, output)
	model.summary()

	return model


def_path_audio_val = '/home/rick/Documents/image_in_audio_steganography/all_stft_val/'
def_path_images = '/home/rick/Documents/image_in_audio_steganography/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
def_audio_path_train = '/home/rick/Documents/image_in_audio_steganography/libri_stft/'

data_audio = glob.glob(def_audio_path_train+'*')
data_audio_val = glob.glob(def_path_audio_val +'*')
data_image = glob.glob(def_path_images+'*')

config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95))
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

tf.keras.backend.set_session(session=sess)

steg_model = steg_model_v3.create_steganography_model()
model = decoder_model.create_decoder_model(steg_model)

model.load_weights('/home/rick/Documents/image_in_audio_steganography/Steg/weights/version_4_UNET19')

denoiser = denoiser_model()
denoiser.load_weights('denoiser_weights/16')
#pred_gen = DataGenerator(data_audio, data_image, 1, def_audio_path=def_audio_path_train)

finc = 0
i = 0

for filename in os.listdir('/home/rick/Documents/image_in_audio_steganography/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'):
	img = cv2.imread('/home/rick/Documents/image_in_audio_steganography/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'+filename)
	img = cv2.resize(img, (513, 512))
	image_out = img
	cv2.imwrite('predictions/clean/'+str(i)+'.png', cv2.resize(img, (256,256)))
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for j in range(3):
		image = img[:,:,j]
		audio = np.load(def_audio_path_train+str(finc+1)+'.npy')
		audio = np.reshape(audio, (512, 513, 1))
		finc = (finc+1)%178
		abs_audio = np.zeros((1,512,513,1))
		abs_audio[:,:512,:,:] = np.abs(audio)
		image = np.reshape(image, (1, 512, 513, 1)) 

		pred = model.predict([image/255.0, abs_audio])
		#print(image_out.shape)
		image_out[:,:,j] = (pred[1][0]*255).astype(np.uint8)[:,:,0]
		
	image_out = cv2.resize(image_out, (256,256))
	image_out = image_out/255.0
	image_out = np.expand_dims(image_out, axis=0)
	image_out = denoiser.predict(image_out)

	cv2.imwrite('predictions/dirty/'+str(i)+'.png', (image_out[0]*255).astype(np.uint8))
	i+=1
	print(i)
	if(i==30):
		break