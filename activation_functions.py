import numpy as np

def sign(x):
	return np.where(np.array(x)>=0,1,-1)

def tanh(x):
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def ELU(x,alpha=0.0001):
	return np.where(x>0,x,alpha*(np.exp(x)-1))

def ReLU(x):
	return np.where(x>0,x,0)

def LeakyReLU(x,alpha=0.0001):
	return np.where(x>0,x,alpha*x)

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x),axis=0)