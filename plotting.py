import numpy as np
import matplotlib.pyplot as plt


def plot_implicit(W):
	a,b,c = W
	x = np.array((plt.xlim()))
	y = eval('-a*x/b-c/b')
	plt.plot(x,y,c='black')

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

	Z = np.where(np.array(Z)>=0,1,-1)

	plt.pcolormesh(xx,yy,Z,cmap=cmap_light)#,norm=norm)
	
def plot_data(X,y=None):
	if(y is None):
		plt.scatter(X[:,0],X[:,1])
	else:
		pos = X[np.where(y==1)]
		neg = X[np.where(y==-1)]
		plt.scatter(pos[:,0],pos[:,1],c='blue',marker='o')
		plt.scatter(neg[:,0],neg[:,1],c='red',marker='x')
	plt.ylim(0,X[:,1].max()+1)
	plt.xlim(0,X[:,0].max()+1)
