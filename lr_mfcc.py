import re
import numpy as np
from scipy.io import wavfile
from sklearn import preprocessing
from python_speech_features import mfcc
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


def load_data(maxword):
	Xtrain = []
	ytrain = []
	Xdevel = []
	ydevel = []
	with open('lab/ComParE2016_Deception.tsv') as f:
		for line in f.readlines()[1: ]:
			lists = line.split(',')
			filepath = 'wav/' + lists[0]
			sampling_freq, audio = wavfile.read(filepath)
			mfcc_features = mfcc(audio, sampling_freq)
			mfcc_features = mfcc_features.reshape(1, mfcc_features.size)
			if re.match('train', lists[0]):
				Xtrain.extend(mfcc_features)
				if re.match('D', lists[1]):
					ytrain.append(1)
				else:
					ytrain.append(0)
			else:
				Xdevel.extend(mfcc_features)
				if re.match('D', lists[1]):
					ydevel.append(1)
				else:
					ydevel.append(0)
	Xtrain = sequence.pad_sequences(Xtrain, maxlen = maxword)
	Xdevel = sequence.pad_sequences(Xdevel, maxlen = maxword)
	return np.array(Xtrain), np.array(ytrain), np.array(Xdevel), np.array(ydevel)

def train(Xtrain, ytrain):	
	model = LogisticRegression()
	model.fit(Xtrain, ytrain)
	return model
	
def test(model, Xtrain, ytrain, Xdevel, ydevel):
	ypred = model.predict(Xtrain)
	print("Model performance on train dataset")
	print(classification_report(ytrain, ypred))
	ypred = model.predict(Xdevel)
	print("Model performance on development dataset")
	print(classification_report(ydevel, ypred))
	
Xtrain, ytrain, Xdevel, ydevel = load_data(760*13)
scaler = preprocessing.StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xdevel = scaler.fit_transform(Xdevel)
model = train(Xtrain, ytrain)
test(model, Xtrain, ytrain, Xdevel, ydevel)
