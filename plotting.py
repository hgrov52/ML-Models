import numpy as np
import matplotlib.pyplot as plt

def set_limits(X):
	plt.ylim(X[:,1].min()-X[:,1].ptp()/10,X[:,1].max()+X[:,1].ptp()/10)
	plt.xlim(X[:,0].min()-X[:,0].ptp()/10,X[:,0].max()+X[:,0].ptp()/10)


def plot_implicit(W,q=1):
	a,b,c = W
	
	x = np.linspace(*plt.xlim(), 100)
	y = np.linspace(*plt.ylim(), 100)
	X, Y = np.meshgrid(x,y)
	F = a*(X**q) + b*(Y**q) + c
	plt.contour(X,Y,F,[0])

# qth order
def plot_mesh(W,X,q=1):
	from matplotlib.colors import Normalize
	from matplotlib.colors import ListedColormap
	cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])#,'#98FB98'])
	cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

	set_limits(X)

	h = .02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
	a,b,c=W
	Z = a*(xx**q)+b*(yy**q)+c

	# not needed when Z is trnasoformed to [-1,1]
	#norm = Normalize(np.min(X),np.max(X))

	Z = np.where(np.array(Z)>=0,1,-1)

	plt.pcolormesh(xx,yy,Z,cmap=cmap_light)#,norm=norm)
	
def plot_data(X,y=None):
	set_limits(X)

	if(y is None):
		plt.scatter(X[:,0],X[:,1])
	else:
		pos = X[np.where(y==1)]
		neg = X[np.where(y==-1)]
		plt.scatter(pos[:,0],pos[:,1],c='blue',marker='o')
		plt.scatter(neg[:,0],neg[:,1],c='red',marker='x')
	
def show():
	plt.show()
