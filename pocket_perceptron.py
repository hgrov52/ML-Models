import numpy as np
import matplotlib.pyplot as plt	
import activation_functions as af
import data,plotting


def perceptron(X,y,max_iter=1000,compare=False):

	# =================================================
	# Iterative method

	i_iterations = 0
	W = np.zeros(X.shape[1]).reshape((1,X.shape[1]))
	
	error = np.inf
	prev_error = np.inf
	best_error = np.inf
	best_weights = None
	misclassified = None
	decreasing_error = True

	while(is_misclassified(W,X,y) and (i_iterations<max_iter or decreasing_error)):
		prev_error = error
		i_iterations += 1
		x,y_temp = get_one_misclassified(W,X,y)
		s = y_temp*x
		W = W+s
		misclassified = calculate_error(W,X,y)
		error = len(misclassified)
		if(error>prev_error):
			decreasing_error = False
		if(error<best_error):
			best_error = error
			best_weights = W
	print(best_error,"/",len(X))

	# =================================================
	# Vectorized method

	# v_start = time.time()
	# v_iterations = 0
	# W2 = np.zeros(X.shape[1]).reshape((1,X.shape[1]))
	# decreasing_error = True
	# while(is_misclassified(W2,X,y)):
	# 	v_iterations+=1
	# 	X_m,y_m = get_all_misclassified(W2,X,y)
	# 	X_m=X_m*y_m.reshape((y_m.shape[0],1))
	# 	W2 = W2+np.sum(X_m,axis=0)
	# v_time = time.time()-v_start	
	
	best_weights = best_weights[0]

	return best_weights,best_error

def classify(W,x):
	return af.sign(np.dot(W,x.T))

def calculate_error(W,X,y):
	misclassified = []
	# iterative
	for i in range(len(X)):
		y_hat = classify(W,X[i])
		if(y_hat!=y[i]):
			misclassified.append(X[i])
	return np.array(misclassified)


def get_one_misclassified(W,X,y):
	misclassified=[]
	for i in range(len(X)):
		if(classify(W,X[i])!=y[i]):
			return X[i],y[i]

def get_all_misclassified(W,X,y):
	misclassified=[]
	y_temp = []
	for i in range(len(X)):
		if(classify(W,X[i])!=y[i]):
			misclassified.append(X[i])
			y_temp.append(y[i])
	return np.array(misclassified),np.array(y_temp)

def is_misclassified(W,X,y):
	for i in range(len(X)):
		if(classify(W,X[i])!=y[i]):
			return True
	return False

def vector_vs_iterative(f,n,d):
	N = np.arange(n)[2:]
	v_time = []
	i_time = []
	v_iterations = []
	i_iterations = []
	for n in N:
		X,y = generate_data(f,n,d)
		W,stats = perceptron(X,y)
		v_iterations.append(stats[0])
		i_iterations.append(stats[1])
		v_time.append(stats[2])
		i_time.append(stats[3])


	from scipy.ndimage.filters import gaussian_filter1d
	v_time = gaussian_filter1d(v_time,sigma=2)
	i_time = gaussian_filter1d(i_time,sigma=2)

	plt.plot(N,v_time,label='vector')
	plt.plot(N,i_time,label='iterative')
	plt.xlabel("Number of data points")
	plt.ylabel("Time (s)")
	plt.ylim(-.0001,min(np.max(v_time),np.max(i_time)))
	plt.legend()
	plt.title("Comparison of Vectorized versus Iterative time")
	plt.show()

	v_iterations = gaussian_filter1d(v_iterations,sigma=2)
	i_iterations = gaussian_filter1d(i_iterations,sigma=2)

	plt.plot(N,v_iterations,label='vector')
	plt.plot(N,i_iterations,label='iterative')
	plt.xlabel("Number of data points")
	plt.ylabel("Number of iterations")
	plt.ylim(-.0001,min(np.max(v_iterations),np.max(i_iterations)))
	plt.legend()
	plt.title("Comparison of Vectorized versus Iterative number of iterations")
	plt.show()
	
if __name__ == '__main__':
	d=2
	n=500
	f = np.random.rand(d+1)
	f[1]*=-1

	#vector_vs_iterative(f,n,d)
	
	X,y = data.generate_linear_data(f,n,d,classify,separable=False)

	W,error = perceptron(X,y)
	print("W:",W)

	plotting.plot_mesh(W,X)
	plotting.plot_implicit(W)
	plotting.plot_data(X,y=y)
	a,b,c=W
	plt.title("Pocket Perceptron\nSlope: {:.2f} | Intercept: {:.2f} | Error: {}/{}".format(-a/b,-c/b,error,len(X)))
	plt.show()


