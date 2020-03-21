import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, Lambda,BatchNormalization, Activation, Dropout, Flatten, Dense, ZeroPadding2D, Input
import numpy as np


def create_model(input_img):


	conv1 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_img)
	conv2 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv1)
	pool1 = keras.layers.MaxPooling2D()(conv2)  #256,256
	print(pool1)
	conv1_new = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(pool1)
	conv2_new = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv1_new)
	pool1_new = keras.layers.MaxPooling2D()(conv2_new)  #128,128
	print(pool1_new)  
	
	conv3 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(pool1_new)
	conv4 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(conv3)
	pool2 = keras.layers.MaxPooling2D()(conv4) #64,64
	print(pool2) #64,64


	conv5 = keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(pool2)

	upsample1 = keras.layers.UpSampling2D()(conv5)
	conv6 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(upsample1)
	conv7 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(conv6)
	print(conv7) #128

	c7_merge = keras.layers.Conv2D(128, (1,1), padding='same')(conv7)
	c4_merge = keras.layers.Conv2D(128, (1,1), padding='same')(conv4)
	merge1 =keras.layers.Add()([c7_merge, c4_merge])
	act1 = keras.layers.Activation('relu')(merge1)

	upsample2 = keras.layers.UpSampling2D()(act1)
	conv8 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(upsample2)
	conv9 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv8)

	c2_merge = keras.layers.Conv2D(64, (1,1), padding='same')(conv2_new)
	c9_merge = keras.layers.Conv2D(64, (1,1), padding='same')(conv9)
	merge2 = keras.layers.Add()([c2_merge, c9_merge])
	act2 = keras.layers.Activation('relu')(merge2)

	upsample3 = keras.layers.UpSampling2D()(act2)
	conv10 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(upsample3)
	conv11 = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv10)
	zero_pad = keras.layers.ZeroPadding2D(((0,0),(1,0)))(conv11)

	zero_merge = keras.layers.Conv2D(64, (1,1), padding='same', activation='relu')(zero_pad)
	c2_merge_old = keras.layers.Conv2D(64, (1,1), padding='same', activation='relu')(conv2)
	merge3 = keras.layers.Add()([c2_merge_old, zero_merge])

	extra_conv = keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(merge3)

	output = keras.layers.Conv2D(3, (3,3), padding='same', activation='sigmoid',  name='img_output')(extra_conv)
	print(output)
	return output

def create_decoder_model(steg_model):
	
	input_layer = Conv2D(64, kernel_size=(3,3), strides=(1,1), name = 'input_decoder', padding='same')(steg_model.output[1])
	act1 = Activation('relu')(input_layer)
	output = create_model(act1)
	model = tf.keras.models.Model(steg_model.input,[steg_model.output[0], output, steg_model.output[2]])
	model.summary()
	keras.utils.plot_model(model, 'model.png', show_shapes='True')
	return model
