import data,plotting
import numpy as np
import pocket_perceptron
import linear_regression

q=6
q_fit=6
noise=0.5
title='data order={} | noise={} | fitting order={}'.format(q,noise,q_fit)
# ======================================
# shell and circle example
# X,y = data.simple_2d(n=50)
# Z = X**q
# Z = data.add_bias(Z)
# W,stats = pocket_perceptron.perceptron(Z,y)
# print(W)
# plotting.plot_mesh(W,X,q=q)
# plotting.plot_data(X,y=y,title=title)
# plotting.plot_transform(W,q=q)
# plotting.show()

# =======================================
# graphing testing
X,f = data.qth_order_data(lower=-8000,upper=10,q=5,n=100000)
z_ = data.zoom_limits(X)
print(z_)
X = data.increase_resolution(f,z_)
plotting.plot_data(X)
plotting.set_limits(z_)
plotting.show()

# ========================================
# linear fit testing
# X,f = data.qth_order_data(q=q,n=50,noise=noise)
# plotting.plot_data(X)
# # returns function

# p = linear_regression.linear_regression(X,q=q_fit)
# plotting.polynomial(p,X)
# print(p.error())

# z_ = data.zoom_limits(X)
# print(z_)
# plotting.plot_data(z_[0],c='r',title=title)

# plotting.show()