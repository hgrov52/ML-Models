import numpy as np

# d-dimensional data
def target_function(d=2):
	#return (np.random.rand(d+1)-0.5)*10
	f = [0.1,0.5,-2,-2,4,0,0,0]
	return f

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
def qth_order_data(lower=-1,upper=1,q=1,n=1000,f=None):
	if(f==None):
		f = target_function(d=4)

	x = np.linspace(lower,upper,n)
	s = ''
	for i,w in enumerate(f):
		s+='+'+str(w)+'*(x**'+str(len(f)-i-1)+')'
	print(s)
	y = eval(s)
	return np.vstack((x,y)).T,f

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

def zoom_limits(X):
	y = X[:,1]
	
	min_indices = np.where(np.r_[1, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:], 1])
	max_indices = np.where(np.r_[1, y[1:] > y[:-1]] & np.r_[y[:-1] > y[1:], 1])
	
	xmin = X[min_indices,0][0]
	xmax = X[max_indices,0][0]

	# print(xmin,y[min_indices])
	# print(xmax,y[max_indices])

	if(len(xmin)<2 or len(xmax)<2):
		return np.array(
			[[X[:,1].min()-X[:,1].ptp()/10,X[:,0].min()-X[:,0].ptp()/10],
			 [X[:,1].max()+X[:,1].ptp()/10,X[:,0].max()+X[:,0].ptp()/10]])

	x_min = np.argmin(xmax)
	x_max = np.argmax(xmin)
	y_min = y[min_indices[0][x_max]]
	y_max = y[max_indices[0][x_min]]
	x_max = xmin[x_max]
	x_min = xmax[x_min]
	
	return np.array([[x_min,y_min],
				  [x_max,y_max]])

# f  -> target function
# z_ -> zoom limits
def increase_resolution(f,z_):
	return qth_order_data(lower=z_[0,0],upper=z_[1,0],f=f,n=10000)[0]

