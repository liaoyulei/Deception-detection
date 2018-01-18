
import re
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc

def load_data():
	XDtrain = []
	XNDtrain = []
	with open('lab/ComParE2016_Deception.tsv') as f:
		for line in f.readlines()[1:]:
			lists = line.split(',')
			if re.match('train', lists[0]):
				filepath = 'wav/' + lists[0]
				sampling_freq, audio = wavfile.read(filepath)
				mfcc_features = mfcc(audio, sampling_freq)
				if re.match('D', lists[1]):
					XDtrain.extend(mfcc_features)
				else:
					XNDtrain.extend(mfcc_features)
				
	return np.array(XDtrain), np.array(XNDtrain)
	
def train(X):
	model = hmm.GaussianHMM(n_components = 4, covariance_type = 'diag', n_iter = 1000)
	model.fit(X)
	return model
	
def test(modelD, modelND):
	tp = 0
	np = 0
	with open('lab/ComParE2016_Deception.tsv') as f:
		for line in f.readlines()[1:]:
			lists = line.split(',')
			if re.match('train', lists[0]):
				filepath = 'wav/' + lists[0]
				sampling_freq, audio = wavfile.read(filepath)
				mfcc_features = mfcc(audio, sampling_freq)
				scoreD = modelD.score(mfcc_features)
				scoreND = modelND.score(mfcc_features)
				if re.match('D', lists[1]) and scoreD >= scoreND:
					tp += 1
				elif re.match('ND', lists[1]) and scoreND >= scoreD:
					tp += 1
				else:
					np += 1
	ans1 = tp / (tp + np)
	tp = 0
	np = 0
	with open('lab/ComParE2016_Deception.tsv') as f:
		for line in f.readlines()[1:]:
			lists = line.split(',')
			if re.match('devel', lists[0]):
				filepath = 'wav/' + lists[0]
				sampling_freq, audio = wavfile.read(filepath)
				mfcc_features = mfcc(audio, sampling_freq)
				scoreD = modelD.score(mfcc_features)
				scoreND = modelND.score(mfcc_features)
				if re.match('D', lists[1]) and scoreD >= scoreND:
					tp += 1
				elif re.match('ND', lists[1]) and scoreND >= scoreD:
					tp += 1
				else:
					np += 1
	ans2 = tp / (tp + np)
	print("Model performance on train dataset")
	print(ans1)
	print("Model performance on devel dataset")
	print(ans2)
					
XDtrain, XNDtrain = load_data()
modelD = train(XDtrain)
modelND = train(XNDtrain)
test(modelD, modelND)
