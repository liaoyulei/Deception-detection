import re
import numpy as np
from scipy.io import wavfile
from sklearn import preprocessing
from python_speech_features import mfcc
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

def load_data():
	Xtrain = []
	ytrain = []
	Xdevel = []
	ydevel = []
	with open('arff/ComParE2016_Deception.ComParE.train.arff') as f:
		for line in f.readlines():
			if re.match('\'', line):
				lists = line.split(',')
				x = np.array(lists[1: -1])
				x = x.reshape(1, x.size)
				Xtrain.extend(x)
				if re.match('D', lists[-1]):
					ytrain.append(1)
				else:
					ytrain.append(0)
	with open('arff/ComParE2016_Deception.ComParE.devel.arff') as f:
		for line in f.readlines():
			if re.match('\'', line):
				lists = line.split(',')
				x = np.array(lists[1: -1])
				x = x.reshape(1, x.size)
				Xdevel.extend(x)
				if re.match('D', lists[-1]):
					ydevel.append(1)
				else:
					ydevel.append(0)
	return np.array(Xtrain), np.array(ytrain), np.array(Xdevel), np.array(ydevel)
	
def train(Xtrain, ytrain, Xdevel, ydevel, maxword):
	model = Sequential()
	model.add(Dense(2048, input_shape = (maxword, ), activation = 'relu'))
	model.add(Dense(512, activation = 'relu'))
	model.add(Dense(128, activation = 'relu'))
	model.add(Dense(32, activation = 'relu'))
	model.add(Dense(1, activation = 'sigmoid'))
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'recall'])
	model.fit(Xtrain, ytrain, validation_data = (Xdevel, ydevel), epochs = 10, batch_size = 143, verbose = 1, class_weight = 'balanced')
	return model
	
def test(model, Xtrain, ytrain, Xdevel, ydevel):
	print(model.summary())
	score = model.evaluate(Xtrain, ytrain)
	print("Model performance on train dataset")
	print(score)
	print("Model performance on development dataset")
	score = model.evaluate(Xdevel, ydevel)
	print(score)

Xtrain, ytrain, Xdevel, ydevel = load_data()
scaler = preprocessing.StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xdevel = scaler.fit_transform(Xdevel)
model = train(Xtrain, ytrain, Xdevel, ydevel, Xtrain.shape[1])
test(model, Xtrain, ytrain, Xdevel, ydevel)
