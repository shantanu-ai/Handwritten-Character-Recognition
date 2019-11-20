import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from skimage.transform import resize

LABELS_PATH = "finalLabelsTrain.npy"
DATA_PATH = "train_data.pkl"
CHAR_LIST = ['a','b','c','d','h','i','j','k']

def load_pkl(train_data_name):
	with open(train_data_name,'rb') as f:
		return pickle.load(f)

def getData(data_path, labels_path):
	DATA = load_pkl(data_path)
	LABELS = np.load(labels_path)
	DATASET = pd.DataFrame({'label':LABELS,'images': DATA}, columns=['label','images'])
	#DATASET = DATASET[DATASET.images == [x for x in np.array(DATASET.images) if not x.shape==()]]
	#print(DATASET)
	char_dict = {}	
	dataset = {}
	test_data = []
	for i in range(0,len(CHAR_LIST)):
		c = CHAR_LIST[i]
		train_data_name = c+'_train'
		test_data_name = c+'_test'
		dataset_name = CHAR_LIST[i]+'_DATASET'
		char_dict[dataset_name] = DATASET[DATASET.label == i+1]
		data = DATASET[DATASET.label == i+1]
		msk = np.random.rand(len(data)) < 0.8
		train = data[msk]
		test = data[~msk]
		dataset[train_data_name] = train
		#dataset[test_data_name] = test
		test_data.append(np.array(test.images))
	'''
	char_dict[a_DATASET] = DATASET[DATASET.label == 1]
	char_dict[b_DATASET] = DATASET[DATASET.label == 2]
	char_dict[c_DATASET] = DATASET[DATASET.label == 3]
	char_dict[d_DATASET] = DATASET[DATASET.label == 4]
	char_dict[h_DATASET] = DATASET[DATASET.label == 5]
	char_dict[i_DATASET] = DATASET[DATASET.label == 6]
	char_dict[j_DATASET] = DATASET[DATASET.label == 7]
	char_dict[k_DATASET] = DATASET[DATASET.label == 8]

	char_list = ['a','b','c','d','h','i','j','k']
	dataset = {}
	for c in char_list:
		train_data_name = c+'_train'
		test_data_name = c+'_test'
		dataset_name = c+'_DATASET'
	msk = np.random.rand(len(a_DATASET)) < 0.8
	a_train = a_DATASET[msk]
	a_test = a_DATASET[~msk]

	msk = np.random.rand(len(b_DATASET)) < 0.8
	b_train = b_DATASET[msk]
	b_test = b_DATASET[~msk]

	X_train = np.array(train_data.images)
	Y_train = np.array(train_data.label)
	X_test = np.array(test_data.images)
	Y_test = np.array(test_data.label)
	
	dataset = {
	"a_train":a_train,
	"b_train":b_train,
	"a_test":a_test,
	"b_test":b_test
	}
	'''
	return dataset,np.array(test_data)

def getFlatten(dataset,test_data):
	dataset_flatten = {}
	for c in CHAR_LIST:
		train_data_name = c+'_train'
		train_data = np.array([resize(np.array(x),(50,50)) for x in np.array(dataset[train_data_name].images)])
		print(train_data.shape)
		train_flatten = train_data.reshape(train_data.shape[0],-1).T
		dataset_flatten[train_data_name] = train_flatten

	test = np.array([resize(np.array(t),(50,50)) for y in test_data for t in y])
	test_flatten = test.reshape(test.shape[0],-1).T
	return dataset_flatten,test_flatten

def sigmoid(X):
	A = 1/(1+np.exp(-X))
	return A

def initialize_parameters(n_x,n_h,n_y):
	W1 = np.random.randn(n_h,n_x)
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h)
	b2 = np.zeros((n_y,1))

	parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
	return parameters

def forward_propogation(X, parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	Z1 = np.dot(W1,X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)

	cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
	return A2, cache

def compute_cost_function(A2,Y,parameters):
	m = Y.shape[1]
	logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
	cost = -np.sum(logprobs)/m
	cost = float(np.squeeze(cost))
	return cost

def backward_propogation(parameters,cache,X,Y):
	m=X.shape[1]

	W1 = parameters["W1"]
	W2 = parameters["W2"]

	A1 = cache["A1"]
	A2 = cache["A2"]

	dZ2 = A2-Y
	dW2 = np.dot(dZ2,A1.T)/m
	db2 = np.sum(dZ2, axis=1, keepdims=True)/m
	dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
	dW1 = np.dot(dZ1,X.T)
	db1 = np.sum(dZ1, axis=1, keepdims=True)/m

	grads = {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}
	return grads

def update_parameters(parameters, grads, learning_rate=0.1):
	W1 = parameters["W1"] - learning_rate*grads["dW1"]
	b1 = parameters["b1"] - learning_rate*grads["db1"]
	W2 = parameters["W2"] - learning_rate*grads["dW2"]
	b2 = parameters["b2"] - learning_rate*grads["db2"]

	parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
	return parameters

def nn_model(X,Y,n_h,iterations):
	n_x = X.shape[0]
	n_y = Y.shape[0]

	parameters = initialize_parameters(n_x,n_h,n_y)
	for i in range(iterations):
		A2,cache = forward_propogation(X,parameters)
		cost = compute_cost_function(A2,Y,parameters)
		grads = backward_propogation(parameters,cache,X,Y)
		parameters = update_parameters(parameters, grads)
	return parameters

def char_model(dataset_flatten,dataset,n_h,iterations=1000):
	char_model_parameters = {}
	for c in CHAR_LIST:
		parameters_name = c+'_parameters'
		dataset_name = c+'_train'
		Y = np.array(dataset[dataset_name].label).reshape(dataset[dataset_name].label.shape[0],1)
		X = dataset_flatten[dataset_name]
		char_model_parameters[parameters_name] = nn_model(X,Y,n_h,iterations)
	return char_model_parameters

def predict(parameters,X):
	A2,cache = forward_propogation(X,parameters)
	#predictions = A2>0.5
	return A2


dataset,test_data = getData(DATA_PATH,LABELS_PATH)
dataset_flatten, test_data_flatten = getFlatten(dataset,test_data)
print(dataset_flatten["a_train"].shape)
print(np.array(dataset["a_train"].label).shape)
print(test_data_flatten.shape)

char_model_parameters = char_model(dataset_flatten,dataset,4)

#[print(y.shape) for y in dataset_flatten["a_train"]]
'''
#X = np.array([resize(x,(50,50)) for x in np.array(dataset["a_train"].images) if not x.shape==()])
dataset,test_data = getData(DATA_PATH,LABELS_PATH)
#print(np.array(dataset["a_train"].images))
a_train = np.array([resize(np.arrya(x),(50,50)) for x in np.array(dataset["a_train"].images)])
#a_test = np.array([resize(x,(50,50)) for x in np.array(dataset["a_test"].images) if x.shape==()])

#plt.imshow(np.array(dataset["a_train"].images)[0])
#a_train_flatten = a_train.reshape(a_train.shape[0],-1).T
#a_test_flatten = a_test.reshape(a_test.shape[0],-1).T

#a_parameters = model(X_flatten, np.array(dataset["a_train"].label), 4)
#a_parameters = model(a_train_flatten, np.ones((1,a_train_flatten.shape[1])), 4)
#print(predict(a_parameters,a_test_flatten))
'''