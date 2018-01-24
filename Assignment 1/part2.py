import tensorflow as tf
import numpy as np
import matplotlib .pyplot as plt


def sqr_distance(X, Z):
	X = tf.expand_dims(X, 1)
	Z = tf.expand_dims(Z, 0)
	T = tf.square(X - Z)
	T = tf.reduce_sum(T, axis=2)
	return T

def knn_responsiblities(s, X, pt, k=2):
	dist = s.run(tf.squeeze(sqr_distance(X, pt)))
	_, idx = s.run(tf.nn.top_k(-1*dist, k=k))
	zeros = s.run(tf.zeros(tf.shape(dist), dtype=tf.float64))
	zeros[idx] = 1.0/k
	return zeros

def knn_matrix(X, pt, Y, k=1):
	dist = sqr_distance(X, pt)
	_, idx = tf.nn.top_k(-1*tf.transpose(dist), k=k)
	return tf.gather(Y, idx)

def compute_mse(X, pt, Y, k=1):
	pts = tf.expand_dims(pt, 1)
	predictY = tf.reduce_mean(knn_matrix(X, pts, Y, k=k), axis=1)
	sqr_err = tf.square(predictY - Y)
	mse = 0.5 * tf.reduce_mean(sqr_err)
	return mse


np.random.seed(521)
data = np.linspace(0, 10, num=100)
target = np.sin(data) + 0.1*np.power(data, 2) + 0.5*np.random.randn(100)
randIdx = np.arange(100)
np.random.shuffle(randIdx)

trainData, trainTarget = data[randIdx[:80]], target[randIdx[:80]]
testData, testTarget = data[randIdx[80:90]], target[randIdx[80:90]]
validData, validTarget = data[randIdx[90:]], target[randIdx[90:]]
trueX = np.linspace(0, 11, num=1000)


with tf.Session() as s:
	X = tf.expand_dims(trainData, 1)

	for k in [1, 3, 5, 50]:
		pts = tf.expand_dims(trueX, 1)
		predictY = tf.reduce_mean(knn_matrix(X, pts, trainTarget, k=k), axis=1)
		predictY = s.run(predictY)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.title('K = %s' % k)
		plt.xlabel('x')
		plt.ylabel('y')
		ax.plot(trueX, predictY, '-', trainData, trainTarget, 'ro')

		# Train
		mse = s.run(compute_mse(X, trainData, trainTarget, k))
		print 'K=%s, TRAIN MSE=%s' % (k, mse)
		# Test
		mse = s.run(compute_mse(X, testData, testTarget, k))
		print 'K=%s, TEST MSE=%s' % (k, mse)
		# Validate
		mse = s.run(compute_mse(X, validData, validTarget, k))
		print 'K=%s, VALIDATE MSE=%s' % (k, mse)
		print '\n'

	plt.show()
