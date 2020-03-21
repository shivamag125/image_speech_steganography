import numpy as np
import tensorflow as tf
from tensorflow import keras
import denoiser_dataloader
from denoiser_tb import tensorboard_custom

def ssim_loss(y_true, y_pred, ratio=0.3):
	mse = tf.keras.losses.MSE(y_true, y_pred)
	ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03))
	return (1.0 - ratio) * (-ssim) + ratio * mse


def create_model(img_shape=(256,256,3)):
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

model = create_model()

train_gen = denoiser_dataloader.DataGenerator(num_imgs=3000)

model.compile(optimizer="adam",
	loss=ssim_loss)

for i in range(300):
	callback_c = tensorboard_custom(model,train_gen)
	model.fit_generator(train_gen, steps_per_epoch=3000//8,epochs=20, callbacks=[callback_c])
	model.save_weights('denoiser_weights/'+str(i))