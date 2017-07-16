import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.05)
    bs = np.arange(-0.5, 0.5, 0.05)

    X, Y = np.meshgrid(ws, bs)

    os = np.array([actfunc(tf.constant(w*i + b)).eval(session=sess) for w,b in zip(np.ravel(X), np.ravel(Y))])

    Z = os.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
    plt.show()

sess = tf.Session();
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
w = tf.random_normal(shape=[3, 3])
b = tf.random_normal(shape=[1, 3])
def func(x): return x
act = func(tf.matmul(i, w) + b)
act.eval(session=sess)

plot_act(1.0, func)
plot_act(1, tf.sigmoid)

act = tf.sigmoid(tf.matmul(i, w) + b)
act.eval(session=sess)

act = tf.tanh(tf.matmul(i, w) + b)
act.eval(session=sess)

plot_act(1, tf.nn.relu)

act = tf.nn.relu(tf.matmul(i, w) + b)
act.eval(session=sess)