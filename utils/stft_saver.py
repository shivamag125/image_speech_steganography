import numpy as np
import tensorflow as tf
from scipy import signal
import pandas as pd
import pprint
import cv2
import glob
import os
import soundfile as sf
def_path = '/home/rick/Documents/image_in_audio_steganography/one_sec_stft/'
# def_path_audio = '/home/rick/Documents/image_in_audio_steganography/libri_audio/'
path_name = '/home/rick/Documents/image_in_audio_steganography/audio_unorder_one_sec/*wav'
waveform = tf.placeholder(tf.float32)
signals = tf.reshape(waveform, [1, -1])
# stfts = tf.contrib.signal.stft(signals, 
#     frame_length=1024, 
#     frame_step=128,
#     fft_length=1024,
#     window_fn=tf.signal.hann_window,
#     pad_end=True)
stfts = tf.contrib.signal.stft(signals, 
    frame_length=1024, 
    frame_step=62, #31 for 1 sec
    fft_length=1024,
    window_fn=tf.signal.hann_window,
    pad_end=True)
sess1 = tf.Session()

def stu(audio_file):
    sample_rate = 16000
    audio_file_1 = audio_file
    #pprint.pprint(def_audio_path)
    #print(audio_file)
    samples,sr = sf.read(audio_file_1)
    # samples = samples/abs(samples.max())
    # samples=audio_file
    print(len(samples))
    #print(sr)
    # print(abs(samples).max())
    # samples = samples/abs(samples).max()
    #print(samples.shape)
    global sess1

    stft = sess1.run([stfts], feed_dict = {waveform:samples})
    stft = np.array(stft)
    stft = np.squeeze(stft)
    print(stft.shape)
    return stft,samples

# a = np.zeros((31744,1))
# stft, audio = stu(a)    
i =0
for file in glob.glob(path_name):
    stft, samples = stu(file)
    i = i+1
    np.save(def_path + str(i) +'.npy', stft)
    # np.save(def_path_audio +str(i)+'.npy', samples)