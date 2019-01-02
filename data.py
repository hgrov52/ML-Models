import numpy as np

def generate_target_function(d=2):
	return 1

def generate_linear_data(f,n,d,classify,max_num=10,separable=True):
	y = []
	X = []
	while(len(y)<n):
		x = np.hstack((np.random.rand(d)*max_num,np.array(1)))
		X.append(x)
		y.append(classify(f,x))
	X = np.array(X)
	y = np.array(y)

	if(not separable):
		# randomly change sign of 5% of the data to make non separable
		indices = np.random.permutation(n)[:n//20]
		y[indices] = -y[indices]

	return X,y

def generate_d_data():
	return 1

def normalize(X):
	return (X-X.min())/(X.max()-X.min())

