import tensorflow as tf
import numpy as np
from data_loader_imgs_saved_stft import DataGenerator
import glob
class tensorboard_custom(tf.keras.callbacks.Callback):
	def __init__(self, model, generator):
		self.seen = 0
		self.index=0
		self.generator = generator
		self.model = model

	def on_batch_end(self, batch, logs={}):
		self.seen += 1
		self.index+=1
		if(self.index>145):
			self.index = 0
		if self.seen % 140 == 0: # every 200 iterations or batches, plot the costumed images using TensorBorad;
			data = self.generator[np.random.randint(0,self.index)]
			inputs = data[0]
			outputs = data[1]
			in_img = inputs[0]
			in_stft = inputs[1]
			out_img = outputs[1]
			out_spec = outputs[0]

			# out_spec = out_spec[:,:512,:,:]

			t_loss = logs.get('loss')
			pred = self.model.predict(inputs, batch_size=1)

			p1 = tf.placeholder(dtype=tf.float32)
			m1 = tf.placeholder(dtype=tf.float32)
			mag_tensor = tf.squeeze(m1, axis=3)
			phasetensor = tf.squeeze(p1, axis=3)
			complex_stft = tf.complex(mag_tensor*tf.math.cos(phasetensor), mag_tensor*tf.math.sin(phasetensor))

			inverse_stft = tf.contrib.signal.inverse_stft(complex_stft, frame_length=1024, 
															frame_step=128,#31 for 1 sec
															window_fn=tf.signal.inverse_stft_window_fn(128),
															fft_length=1024)
			inverse_stft = inverse_stft
			inverse_stft = tf.reshape(inverse_stft, (1,-1,1))
			sess= tf.Session()
			output_audio_pred ,phase= sess.run([inverse_stft, phasetensor], feed_dict = {p1: pred[2], m1:pred[0]})
			# print(output_audio_pred[0].shape)
			# exit()
			c1 = tf.placeholder(dtype=tf.complex64)
			c2 = tf.squeeze(c1, axis =3)
			inverse_stft_input = tf.contrib.signal.inverse_stft(c2, frame_length=1024, 
															frame_step=128,#128 for 4 sec
															window_fn=tf.signal.inverse_stft_window_fn(128),
															fft_length=1024)
			
			inverse_stft_input_1 = inverse_stft_input
			inverse_stft_input_2 = tf.expand_dims(inverse_stft_input_1, axis=2)
			input_audio = sess.run([inverse_stft_input_2], feed_dict={c1:in_stft})
			output_image = pred[1]
			writer = tf.summary.FileWriter('/home/morty/Documents/image_in_audio_steganography/logs_color')
			summary_loss = tf.summary.scalar('train_loss', t_loss)
			summary_image = tf.summary.image('input_img', in_img*255)
			summary_audio = tf.summary.audio('input_audio', input_audio[0], 16000)
			summary_audio_op = tf.summary.audio('output_audio', output_audio_pred, 16000)
			summary_image_op = tf.summary.image('output_img', output_image*255)
			k=tf.summary.merge_all()
			with tf.Session() as sess2:
				b = sess2.run(k)
				writer.add_summary(summary=b, global_step=self.seen)
				writer.close()
			return
