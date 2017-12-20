#!/opt/local/bin/python3

"""
Solving the Poisson equation in 2D
with DNN using TensorFlow 1.2
12/20/2017
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time

##############################
# Problem definition
##############################

eta = 0.0001
maxit = 5000
mini_batch_size = 5

N      = 10   # number of quadrature points in one direction
N2     = N*N  # number of total quadrature points
quad   = np.linspace(-1, 1, N)
x_quad = np.array( [[x1, x2] for x1 in quad for x2 in quad] )

dim = 2   # space dimension for PDE
xpt = []  # quadrature points: x0, x1, ...
for i in range(dim):
    xpt.append(tf.get_variable('xpt'+str(i),
                               shape=[N*N,1],
                               initializer=tf.zeros_initializer,
                               trainable=False))
Xpt = tf.concat(xpt, axis=0)
Xpt = tf.reshape(Xpt, [N*N, dim])

# Form a DNN
M = 6  # number of nodes on each level
L = 4  # number of hidden levels in DNN
layer_structure = [dim, M, M, M, M, 1]
activate = tf.nn.relu

# Define weights and bias
W, b = [], []
for i in range(L+1):
    W.append(tf.get_variable(name='W' + str(i),
                             shape=layer_structure[i:i+2],
                             initializer=tf.truncated_normal_initializer))
for i in range(L):
    b.append(tf.get_variable(name='b' + str(i),
                             shape=[1, layer_structure[i+1]],
                             initializer=tf.truncated_normal_initializer))

# Define network: 1 input, 4 hidden, 1 output
X1 = activate(tf.matmul(Xpt, W[0]) + b[0])
X2 = activate(tf.matmul(X1,  W[1]) + b[1])
X3 = activate(tf.matmul(X2,  W[2]) + b[2])
X4 = activate(tf.matmul(X3,  W[3]) + b[3])

upt    = tf.matmul(X4, W[4])
upt_x0 = tf.gradients(upt, [xpt[0]])
upt_x1 = tf.gradients(upt, [xpt[1]])

##############################
# Deep neural network
##############################

dim = 2 # space dimension for PDE
x = []  # x0, x1, ...
for i in range(dim):
    x.append(tf.get_variable('x'+str(i),
                             initializer=tf.constant([0.0]),
                             trainable=False))
X = tf.concat(x, axis=0)
X = tf.reshape(X, [1, dim])

# Define network: 1 input, 4 hidden, 1 output
X1 = activate(tf.matmul(X,  W[0]) + b[0])
X2 = activate(tf.matmul(X1, W[1]) + b[1])
X3 = activate(tf.matmul(X2, W[2]) + b[2])
X4 = activate(tf.matmul(X3, W[3]) + b[3])
u  = tf.matmul(X4, W[4])

# Get trainable parameters
theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# Compute gradients
u_theta = tf.gradients(u, theta)

u_x0 = tf.gradients(u, [x[0]])
u_x1 = tf.gradients(u, [x[1]])

u_x0_theta = tf.gradients(u_x0, theta)
u_x1_theta = tf.gradients(u_x1, theta)

# Set values for a sampling point
value0 = tf.Variable(0.0 , name='point_0')
value1 = tf.Variable(0.0 , name='point_1')
point0 = tf.assign(x[0], [value0])
point1 = tf.assign(x[1], [value1])

# Update current parameters
update, g = [], []
counter = 0
for t in theta:
    g.append(tf.get_variable(name='gradient'+str(counter), shape=t.get_shape()))
    update.append(tf.assign_add(t, - eta * (1.0 / mini_batch_size) * g[counter]))
    counter += 1 # a counter for collection

# Initialize variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

##############################
# Training algorithm
##############################

time_start = time()

for i in range(maxit):

    # Output iterative infomation
    if not i % 100:
        uval   = sess.run(upt,    feed_dict={Xpt: x_quad})
        ux0val = sess.run(upt_x0, feed_dict={Xpt: x_quad})
        ux1val = sess.run(upt_x1, feed_dict={Xpt: x_quad})

        loss = 0.0
        for ipt in range(N2):
            loss += uval[ipt]**2 + ux0val[0][ipt]**2 + ux1val[0][ipt]**2
        loss /= N2*2
        print('iter =', i, 'objective function value =', loss[0])
        time_now = time()
        print('CPU time used:', time_now-time_start, 'seconds')
        time_start = time_now

    # Compute SGD
    mini_batch_index = np.random.choice(N2, mini_batch_size)
    mini_batch = x_quad[mini_batch_index, :]

    for sample in range(mini_batch_size):

        # sampling point location
        sess.run(point0, feed_dict={value0: mini_batch[sample, 0]})
        sess.run(point1, feed_dict={value1: mini_batch[sample, 1]})

        A0_0,  A0_1  = sess.run([u, u_theta])
        A1_00, A1_01 = sess.run([u_x0, u_x1])
        A1_10, A1_11 = sess.run([u_x0_theta, u_x1_theta])

        if sample == 0:
            gradient = [A1_00[0] * x + A1_01[0] * y + A0_0 * z
                        for x, y, z in zip(A1_10, A1_11, A0_1)]
        else:
            gradient = [A1_00[0] * x + A1_01[0] * y + A0_0 * z + g
                        for g, x, y, z in zip(gradient, A1_10, A1_11, A0_1)]

    # Update parameters using SGD
    for k in range(counter):
        sess.run(update[k], feed_dict={g[k]: gradient[k]})

# Compute function values at quadrature points and plot
ans = sess.run(upt, feed_dict={Xpt: x_quad})
fig = plt.figure()
ax = fig.gca(projection='3d')
xx, yy = np.meshgrid(quad, quad)
ax.plot_surface(xx, yy, np.reshape(ans, [N, N]))
plt.show()

