import re
import numpy as np
from random import shuffle
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, Activation, LSTM

def load_data(maxword):
	Xtrain = []
	ytrain = []
	Xdevel = []
	ydevel = []
	with open('text/vec.txt') as f:
		for line in f.readlines():
			lists = line.split()
			if re.match('train|devel', lists[0]):
				file = lists[0]
				sampling_freq, audio = wavfile.read('wav/' + file)
				features = mfcc(audio, sampling_freq)
				np.append(features, lists[3: ])
				if re.match('D', lists[1]):
					label = 1
				else:
					label = 0
			elif re.match(']', lists[-1]):
				np.append(features, lists[: -1])
				if re.match('train', file):
					Xtrain.extend(features.reshape(1, features.size))
					ytrain.append(label)
				else:
					Xdevel.extend(features.reshape(1, features.size))
					ydevel.append(label)				
			else:
				np.append(features, lists)
	Xtrain = sequence.pad_sequences(Xtrain, maxlen = maxword * 13)
	Xtrain = Xtrain.reshape(Xtrain.shape[0], maxword, 13)
	Xdevel = sequence.pad_sequences(Xdevel, maxlen = maxword * 13)
	Xdevel = Xdevel.reshape(Xdevel.shape[0], maxword, 13)
	return np.array(Xtrain), np.array(ytrain), np.array(Xdevel), np.array(ydevel)
	
def train(Xtrain, ytrain, Xdevel, ydevel, maxword):
	model = Sequential()
	model.add(LSTM(128, input_shape = (maxword, 13, ), return_sequences = True))
	model.add(Dropout(0.2))
	model.add(LSTM(64, return_sequences = True))
	model.add(Dropout(0.2))
	model.add(LSTM(32))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation = 'sigmoid'))
	model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
	print(model.summary())
#	model.fit(Xtrain, ytrain, validation_data = (Xdevel, ydevel), epochs = 10, batch_size = 10, verbose = 1)
	return model
	
def test(model, Xtrain, ytrain, Xdevel, ydevel):
	score = model.evaluate(Xtrain, ytrain)
	print("Model performance on train dataset")
	print(score)
	print("Model performance on development dataset")
	score = model.evaluate(Xdevel, ydevel)
	print(score)

Xtrain, ytrain, Xdevel, ydevel = load_data(800)
index = [i for i in range(Xtrain.shape[0])]    
shuffle(index) 
Xtrain = Xtrain[index]  
ytrain = ytrain[index]
model = train(Xtrain, ytrain, Xdevel, ydevel, 800)
test(model, Xtrain, ytrain, Xdevel, ydevel)
