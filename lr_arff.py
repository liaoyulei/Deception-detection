import re
import numpy as np
from scipy.io import wavfile
from sklearn import preprocessing
from python_speech_features import mfcc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

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
	
Xtrain, ytrain, Xdevel, ydevel = load_data()
scaler = preprocessing.StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xdevel = scaler.fit_transform(Xdevel)
model = train(Xtrain, ytrain)
test(model, Xtrain, ytrain, Xdevel, ydevel)
