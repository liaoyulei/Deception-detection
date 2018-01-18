import re
import numpy as np
from scipy.io import wavfile
from sklearn.cluster import KMeans
from python_speech_features import mfcc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

def bow():
	keypoints_all = []
	with open('lab/ComParE2016_Deception.tsv') as f:
		for line in f.readlines()[1:]:
			lists = line.split(',')
			filepath = 'wav/' + lists[0]
			sampling_freq, audio = wavfile.read(filepath)
			mfcc_features = mfcc(audio, sampling_freq)
			keypoints_all.extend(mfcc_features)
	keypoints_all = np.array(keypoints_all)
	return cluster(keypoints_all)

def cluster(datapoints):
	kmeans = KMeans(init = 'k-means++', n_clusters = 5, n_init = 100)
	res = kmeans.fit(datapoints)
	centroids = res.cluster_centers_
	return kmeans, centroids

def load_data(kmeans, centroids):
	Xtrain = []
	ytrain = []
	Xdevel = []
	ydevel = []
	with open('lab/ComParE2016_Deception.tsv') as f:
		for line in f.readlines()[1:]:
			lists = line.split(',')
			filepath = 'wav/' + lists[0]
			sampling_freq, audio = wavfile.read(filepath)
			mfcc_features = mfcc(audio, sampling_freq)
			mfcc_features = construct_features(mfcc_features, kmeans, centroids)
			if re.match('train', lists[0]):
				Xtrain.extend(mfcc_features)
				ytrain.append(lists[1])
			else:
				Xdevel.extend(mfcc_features)
				ydevel.append(lists[1])
	return np.array(Xtrain), np.array(ytrain), np.array(Xdevel), np.array(ydevel)

def construct_features(mfcc, kmeans, centroids):
	labels = kmeans.predict(mfcc)
	feature_vector = np.zeros(5)
	for i, item in enumerate(feature_vector):
		feature_vector[labels[i]] += 1
	feature_vector = feature_vector.reshape(1, feature_vector.shape[0])
	return normalize(feature_vector)
	
def normalize(input_data):
	sum_input = np.sum(input_data)
	if sum_input > 0:
		return input_data / sum_input
	else:
		return input_data
	
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

kmeans, centroids = bow()
Xtrain, ytrain, Xdevel, ydevel = load_data(kmeans, centroids)
model = train(Xtrain, ytrain)
test(model, Xtrain, ytrain, Xdevel, ydevel)
