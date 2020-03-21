import tensorflow as tf
import numpy as np
import glob
import cv2

class tensorboard_custom(tf.keras.callbacks.Callback):
	def __init__(self, model, generator):
		self.seen = 0
		self.index = 0
		self.generator = generator
		self.model = model

	def on_batch_end(self, batch, logs={}):
		self.seen += 1
		self.index+=1
		if(self.index>100):
			self.index = 0
		if self.seen % 375 == 0: # every 200 iterations or batches, plot the costumed images using TensorBorad;
			data = self.generator[self.index]#######Edit here
			inputs = data[0]
			outputs = data[1]
			cv2.imwrite('../denoising_output/clean.png', cv2.cvtColor((outputs[0]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
			# out_spec = out_spec[:,:512,:,:]

			t_loss = logs.get('loss')
			pred = self.model.predict(inputs, batch_size=1)

			output_image = pred
			cv2.imwrite('../denoising_output/output.png', cv2.cvtColor((output_image[0]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
			writer = tf.summary.FileWriter('/home/rick/Documents/image_in_audio_steganography/logs_denoiser')
			summary_loss = tf.summary.scalar('train_loss', t_loss)
			summary_image = tf.summary.image('input_img', inputs*255)
			summary_image_op = tf.summary.image('output_img', output_image*255)
			k=tf.summary.merge_all()
			with tf.Session() as sess2:
				b = sess2.run(k)
				writer.add_summary(summary=b, global_step=self.seen)
				writer.close()
			return
