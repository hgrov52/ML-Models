import numpy as np
import matplotlib.pyplot as plt	
import time


def perceptron(X,y):

	# =================================================
	# Iterative method

	i_start = time.time()
	i_iterations = 0
	W = np.zeros(X.shape[1]).reshape((1,X.shape[1]))
	while(is_misclassified(W,X,y)):
		i_iterations+=1
		x,y_temp = get_one_misclassified(W,X,y)
		s = y_temp*x
		W = W+s
	i_time = time.time()-i_start

	# =================================================
	# Vectorized method

	v_start = time.time()
	v_iterations = 0
	W2 = np.zeros(X.shape[1]).reshape((1,X.shape[1]))
	while(is_misclassified(W2,X,y)):
		v_iterations+=1
		X_m,y_m = get_all_misclassified(W2,X,y)
		X_m=X_m*y_m.reshape((y_m.shape[0],1))
		W2 = W2+np.sum(X_m,axis=0)
	v_time = time.time()-v_start	

	W=W[0]
	W2=W2[0]

	return W,(v_iterations,i_iterations,v_time,i_time)

def normalize(X):
	return (X-X.min())/(X.max()-X.min())

def sign(n):
	return np.where(np.array(n)>=0,1,-1)

def classify(W,x):
	return sign(np.dot(W,x.T))

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

def generate_separable_data(f,n,d,max_num=10):
	y = []
	X = []
	while(len(y)<n):
		x = np.hstack((np.random.rand(d)*max_num,np.array(1)))
		X.append(x)
		y.append(classify(f,x))
	return np.array(X),np.array(y)

def vector_vs_iterative(f,n,d):
	N = np.arange(n)[2:]
	v_time = []
	i_time = []
	v_iterations = []
	i_iterations = []
	for n in N:
		X,y = generate_separable_data(f,n,d)
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

def plot_implicit(W):
	a,b,c = W
	x = np.array((plt.xlim()))
	y = eval('-a*x/b-c/b')
	plt.plot(x,y,c='black')

	print("slope:",-a/b)
	print("intercept:",-c/b)

def plot_mesh(W,X):
	from matplotlib.colors import Normalize
	from matplotlib.colors import ListedColormap
	cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])#,'#98FB98'])
	cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

	h = .02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
	a,b,c=W
	Z = a*xx+b*yy+c

	# not needed when Z is trnasoformed to [-1,1]
	#norm = Normalize(np.min(X),np.max(X))

	Z = sign(Z)

	plt.pcolormesh(xx,yy,Z,cmap=cmap_light)#,norm=norm)
	
if __name__ == '__main__':
	d=2
	n=100
	f = np.random.rand(d+1)
	f[1]*=-1

	vector_vs_iterative(f,n,d)

	
	X,y = generate_separable_data(f,n,d)

	W,stats = perceptron(X,y)
	print("W:",W)

	plot_mesh(W,X)
	#plt.scatter(X[:,0],X[:,1],c=y,marker='+')
	pos = X[np.where(y==1)]
	neg = X[np.where(y==-1)]
	plt.scatter(pos[:,0],pos[:,1],c='blue',marker='o')
	plt.scatter(neg[:,0],neg[:,1],c='red',marker='x')
	
	
	plt.ylim(0,X[:,1].max()+1)
	plt.xlim(0,X[:,0].max()+1)
	plot_implicit(W)
	a,b,c=W
	plt.title("Ordinary Perceptron\nSlope: {:.2f} | Intercept: {:.2f}".format(-a/b,-c/b))
	plt.show()


