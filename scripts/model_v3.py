import numpy as np
import tensorflow as tf
from tensorflow import keras
import steg_model_v3
import decoder_model
from data_loader_imgs_saved_stft import DataGenerator
import glob
import cv2
import pandas as pd
from tb import tensorboard_custom
import os


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


def ssim_loss(y_true, y_pred, ratio=0.85):
	mae = tf.keras.losses.MSE(y_true, y_pred)
	ssim = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
	return (1.0 - ratio) * (1-ssim) + ratio * mae

def ssim_loss_audio(y_true, y_pred, ratio=0.995):
	mae = tf.keras.losses.MSE(y_true, y_pred)
	ssim = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
	return (1.0 - ratio) * (1-ssim) + ratio * mae

def_path_audio_val = '/home/rick/Documents/image_in_audio_steganography/all_stft_val/'
def_path_images = '/home/morty/Documents/image_in_audio_steganography/data/VOC2012/JPEGImages/'
def_audio_path_train = '/home/morty/Documents/image_in_audio_steganography/one_stft/'

data_audio = glob.glob(def_audio_path_train+'*')
data_audio_val = glob.glob(def_path_audio_val +'*')
data_image = glob.glob(def_path_images+'*')

config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90))
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

tf.keras.backend.set_session(session=sess)

steg_model = steg_model_v3.create_steganography_model()
model = decoder_model.create_decoder_model(steg_model)
# model = multi_gpu_model(model1, gpus=1)

losses = {
	"carrier_output": ssim_loss_audio,
	"img_output": ssim_loss#"mse",
	# ssim_loss
}

lossWeights = {"carrier_output": 0.69, "img_output": 1}

opt = tf.keras.optimizers.SGD(learning_rate=0.0039, momentum=0.85)

model.compile(optimizer="adam",
	loss=losses, 
	loss_weights=lossWeights)


train_gen = DataGenerator(data_audio, data_image, 1, def_audio_path=def_audio_path_train)

for i in range(20):
	callback_c = tensorboard_custom(model,train_gen)
	model.fit_generator(
		generator=train_gen,
		steps_per_epoch=150,
		# validation_data =train_gen,
		epochs=10,callbacks=[callback_c]
		)
	model.save_weights('/home/morty/Documents/image_in_audio_steganography/weights_color/version_4_UNET'+str(i))

pred = model.predict_generator(train_gen,
	steps=10
	)