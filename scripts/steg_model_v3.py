import numpy as np
import tensorflow as tf
from tensorflow import keras

def gated_convolution_block(input_tensor, filters, kernel_size, stride, padding='same', pool=False, name=None):
	conv = keras.layers.Conv2D(filters, kernel_size, stride, padding=padding)(input_tensor)
	gate_conv = keras.layers.Conv2D(filters, kernel_size, stride, padding=padding,name = name)(input_tensor)
	# bn = keras.layers.BatchNormalization()(conv)
	gate = keras.layers.Activation('sigmoid')(gate_conv)
	output = keras.layers.Multiply()([conv, gate])
	if(pool):
		pooled = keras.layers.MaxPool2D()(output)
		return pooled

	return output
	
def cnn_block(input_tensor, filters, kernel_size, stride, padding='same', pool=False, name = None, activation=True):
	if(activation==False):
		conv = keras.layers.Conv2D(filters, kernel_size, stride, name=name, padding=padding)(input_tensor)
		return conv
	conv = keras.layers.Conv2D(filters, kernel_size, stride, padding=padding)(input_tensor)
	bn = keras.layers.BatchNormalization()(conv)
	act = keras.layers.Activation('relu', name = name)(bn)
	if(pool):
		pooled = keras.layers.MaxPool2D()(act)
		return pooled
	return act
	

def message_encoder(message):
	conv1 = cnn_block(message, 128, (7,7), 1, pool=False)
	conv2 = cnn_block(conv1, 128, (5,5), 1, pool=False)
	conv3 = cnn_block(conv2, 64, (3,3), 1, pool=False)
	conv4 = cnn_block(conv3, 64, (3,3), 1, pool=False)

	return conv4

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
	output = keras.layers.Conv2D(1, (3,3), padding='same', name='carrier_output')(extra_conv)
	
	print(output)
	return output


def carrier_encoder(carrier_spectrogram):
	# padded1 = keras.layers.ZeroPadding2D(((1,0),(0,0)))(carrier_spectrogram)
	gated_conv1 = gated_convolution_block(carrier_spectrogram, 128, (3,3), 1, padding='same', name ='carrier_encoder_1')
	gated_conv2 = gated_convolution_block(gated_conv1, 64, (3,3), 1, padding='same',name ='carrier_encoder_2')
	gated_conv3 = gated_convolution_block(gated_conv2, 64, (3,3), 1, padding='same', name ='carrier_encoder_3')

	return [gated_conv2, gated_conv3]

def merged_encoder(merged_tensor):
	gated_conv1 = gated_convolution_block(merged_tensor, 128, (5,5), 1, padding='same')
	# padded = keras.layers.ZeroPadding2D((1,1))(gated_conv1)
	gated_conv2 = gated_convolution_block(gated_conv1, 64, (3,3), 1, padding='same')

	return [gated_conv1, gated_conv2]

def upsampling_block(input_tensor, filters, residual_tensor=None):
	upsampled = keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
	if(residual_tensor is not None):
		merged_tensor = keras.layers.Concatenate()([upsampled, residual_tensor])
		bn1 = keras.layers.BatchNormalization()(merged_tensor)
		conv1 = cnn_block(bn1, filters, (3,3), 1, padding='same')
		conv2 = cnn_block(conv1, filters, (3,3), 1, padding='same')
		return conv2
	else:
		padded = keras.layers.ZeroPadding2D(((0,0),(0,1)))(upsampled)
		return padded

def merged_decoder(hidden_tensor1):
	gated_conv1 = gated_convolution_block(hidden_tensor1, 64, (3,3), 1, padding='same')
	gated_conv2 = gated_convolution_block(gated_conv1, 32, (3,3), 1, padding='same')
	gated_conv3 = gated_convolution_block(gated_conv2, 16, (3,3), 1, padding='same',)
	conv1 = cnn_block(gated_conv3, 1, (3,3), 1, padding='same', name = 'carrier_output', activation=False)
	return conv1

def get_spectrogram_and_phase(stfts):
	magnitude_spectrograms = tf.abs(stfts)
	real_part = tf.real(stfts)
	imag_part = tf.imag(stfts)
	phase = tf.math.atan2(imag_part,(real_part+1e-7))

	return magnitude_spectrograms, phase

def istft_stft_block(mag_tensor, phasetensor):
	mag_tensor = tf.squeeze(mag_tensor, axis=3)
	phasetensor = tf.squeeze(phasetensor, axis=3)
	complex_stft = tf.complex(mag_tensor*tf.math.cos(phasetensor), mag_tensor*tf.math.sin(phasetensor))

	inverse_stft = tf.contrib.signal.inverse_stft(complex_stft, frame_length=1024, 
		frame_step=128,
		window_fn=tf.signal.inverse_stft_window_fn(128),
		fft_length=1024)

	stfts = tf.contrib.signal.stft(inverse_stft[:, :65536], 
		frame_length=1024, 
		frame_step=128,
		fft_length=1024,
		window_fn=tf.signal.hann_window,
		pad_end = True)
	stfts = stfts
	magnitude_spectrograms = tf.abs(stfts)
	magnitude_spectrograms = tf.expand_dims(magnitude_spectrograms, axis=3)
	return magnitude_spectrograms

def create_steganography_model(img_shape = (512, 513, 1), wav_shape = (512, 513, 1)):
	message = keras.layers.Input(shape = img_shape)
	carrier = keras.layers.Input(shape = wav_shape, dtype = tf.complex64)

	carrier_spectrogram, phase = tf.keras.layers.Lambda(get_spectrogram_and_phase)(carrier)

	encoded_carrier_tensors = carrier_encoder(carrier_spectrogram)
	encoded_message_tensor = message_encoder(message)

	merged_tensor = keras.layers.Concatenate()([encoded_carrier_tensors[1], encoded_message_tensor])

	u_net = create_model(merged_tensor)
	model = keras.Model(inputs=[message, carrier], outputs=[u_net, u_net, phase])
	return model