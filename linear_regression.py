import numpy as np
import data,plotting

def magnitude(D):
	return np.sum(np.square(D))**(0.5)

def linear_regression(X,q=1):
	y = X[:,1]
	x = X[:,0]

	X = np.ones(len(x)).reshape((len(x),1))

	for i in range(1,q+1):
		X = np.hstack((X,(x**i).reshape((len(x),1))))
		
	z = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
	return data.Polynomial(z[::-1],X)

def train_SGA(data,epsilon,eta):
	w = np.zeros(data.shape[1]-1)
	iteration=0
	previous_w = np.ones(w.shape)
	while(magnitude(w - previous_w)>epsilon):
		iteration+=1
		previous_w = w.copy()
		# iterate randomly through the entire data set
		for i in np.random.permutation(len(data)):
			# compute gradient at xi
			# θ(Z) = sigmoid = 1/(1+e^-Z) = 1/(1+np.exp(-Z))
			# ▽(w,i) = (yi - θ(wt.T*xi))xi
			xi = data[i][:-1]
			yi = data[i][data.shape[1]-1]
			#print("w shape:",w.shape,"xi shape:",xi.shape,"xi:",xi)
			wTdotx = np.dot(w,xi)
			#print("w.T dot xi:",wTdotx)
			sigmoid = 1/(1+np.exp(-wTdotx))
			#print(sigmoid)
			gradient = np.dot(yi - sigmoid,xi)
			# update the estimator w
			w += eta*gradient

	print(iteration,"iterations")

		
	print("w:",w)
	return w

def test_SGA(data,w):
	num_correct_predictions = 0
	for i in range(len(data)):
		zi = data[i][:-1]
		yi = data[i][data.shape[1]-1] 
		wTdotx = np.dot(w,zi)
		# calculate sigmoid
		sigmoid = 1/(1+np.exp(-wTdotx))
		prediction = 1 if sigmoid >= 0.5 else 0
		if(prediction==yi):
			num_correct_predictions+=1
	return num_correct_predictions/len(data)

if __name__ == '__main__':
	epsilon = .01
	eta = .01

	X = data.qth_order_data(lower=-100,upper=100)

	w = train_SGA(X,epsilon,eta)

	accuracy = test_SGA(test,w)

	print("Accuracy: {:.1f}%".format(accuracy*100))