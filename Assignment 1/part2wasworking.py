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



np.random.seed(521)
data = np.linspace(0, 10, num=100)
target = np.sin(data) + 0.1*np.power(data, 2) + 0.5*np.random.randn(100)
randIdx = np.arange(100)
np.random.shuffle(randIdx)

trainData, trainTarget = data[randIdx[:80]], target[randIdx[:80]]
testData, testTarget = data[randIdx[80:90]], target[randIdx[80:90]]
validData, validTarget = data[randIdx[90:]], target[randIdx[90:]]
trueX = np.linspace(0, 11, num=1000)

all_knn_res = []
k = 3
X = tf.expand_dims(trainData, 1)

# plt.plot(trainData, trainTarget, 'ro')
# plt.show()

with tf.Session() as s:
	for pt in trueX:
		pt = tf.expand_dims(tf.expand_dims(pt, 0), 0)
		knn_r = knn_responsiblities(s, X, pt, k)
		all_knn_res += [knn_r]

	all_knn_res = tf.transpose(np.array(all_knn_res))
	trueY = tf.expand_dims(trainTarget, 0)
	predictY = s.run(tf.squeeze(tf.matmul(trueY, all_knn_res)))

	plt.plot(trueX, predictY, '-', trainData, trainTarget, 'ro')
	plt.show()

	# for pt in trainData:
	# 	pt = tf.expand_dims(tf.expand_dims(pt, 0), 0)
	# 	knn_r = knn_responsiblities(s, X, pt, k)
	# 	all_knn_res += [knn_r]

	# all_knn_res = tf.transpose(np.array(all_knn_res))
	# trueY = tf.expand_dims(trainTarget, 0)

	# sqr_err = tf.square(tf.squeeze(tf.matmul(trueY, all_knn_res) - trueY))
	# mse = 0.5 * tf.reduce_mean(sqr_err)

	# mse = s.run(mse)
	# import pdb
	# pdb.set_trace()
	# pass

