import numpy as np

# d-dimensional data
def target_function(d=2):
	return (np.random.rand(d+1)-0.5)*10

def data(f,n,d,classify,max_num=1,separable=True):
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

# qth-order data
def qth_order_data(lower=-1,upper=1,q=1,n=1000):
	W = target_function(d=4)
	x = np.linspace(lower,upper,n)
	s = ''
	for i,w in enumerate(W):
		s+='+'+str(w)+'*(x**'+str(len(W)-i-1)+')'
	print(s)
	y = eval('x**5+4*x**4')
	return np.vstack((x,y)).T



def shell(radius=2,thickness=1,theta=2*np.pi,n=1000):
	# radius must be strictly bigger
	if(thickness>radius):
		radius=thickness
	theta = np.linspace(0, theta, n)
	r = np.random.uniform(radius-thickness,radius,n)
	X = np.vstack((r * np.cos(theta),r * np.sin(theta))).T
	return X

def circle(radius=1,theta=2*np.pi,n=1000):
	theta = np.linspace(0, theta, n)
	r = np.random.uniform(0,radius,n)
	X = np.vstack((r * np.cos(theta),r * np.sin(theta))).T
	return X

def simple_2d(radius=3,thickness=1,separation=1,n=1000):
	X = np.vstack((shell(radius,thickness,n=n),circle(radius-thickness-separation,n=n)))
	y = np.hstack((np.ones(n),np.zeros(n)-1))
	return X,y

def normalize(X):
	return (X-X.min())/(X.max()-X.min())

def add_bias(X):
	bias = np.ones(X.shape[0]).reshape((X.shape[0],1))
	return np.hstack((X,bias))

