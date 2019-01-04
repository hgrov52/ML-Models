import data,plotting
import numpy as np
import pocket_perceptron

q=2

X,y = data.simple_2d(n=50)
Z = X**q
Z = data.add_bias(Z)


W,stats = pocket_perceptron.perceptron(Z,y)
print(W)


plotting.plot_mesh(W,X,q=q)
plotting.plot_data(X,y=y)
plotting.plot_implicit(W,q=q)

plotting.show()

X = data.qth_order_data(lower=-4,upper=2,q=5)
print(X)
plotting.plot_data(X)
plotting.show()
