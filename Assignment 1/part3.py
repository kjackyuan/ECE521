import tensorflow as tf
import numpy as np

def sqr_distance(X, Z):
	X = tf.expand_dims(X, 1)
	Z = tf.expand_dims(Z, 0)
	T = tf.square(X-Z)
	T = tf.reduce_sum(T, axis=3)
	T = tf.reduce_sum(T, axis=2)
	return T

def knn_predict(s, X, pt, Y, k=1):
	dist = sqr_distance(X, pt)
	_, idx = tf.nn.top_k(-1*tf.transpose(dist), k=k)
	gather = tf.gather(Y, idx)

	count0 = s.run(tf.count_nonzero(gather, axis=1))
	count1 = s.run(tf.count_nonzero(gather-1, axis=1))
	count2 = s.run(tf.count_nonzero(gather-2, axis=1))
	count3 = s.run(tf.count_nonzero(gather-3, axis=1))
	count4 = s.run(tf.count_nonzero(gather-4, axis=1))
	count5 = s.run(tf.count_nonzero(gather-5, axis=1))

	prediction = np.argmin(np.array([count0, count1, count2, count3, count4, count5]), axis=0)
	return prediction

def compute_accuracy(trueY, predictY):
	eq = tf.equal(trueY, predictY)
	accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))
	return accuracy


data = np.load('data.npy').astype(np.int32)
target = np.load('target.npy').astype(np.int32)
target = np.array(map(lambda x: x[1], target))
dataSize = len(data)

np.random.seed(521)
randIdx = np.arange(dataSize)
np.random.shuffle(randIdx)

train_partition = int(dataSize*0.8)
trainData, trainTarget = data[randIdx[:train_partition]], target[randIdx[:train_partition]]
testData, testTarget = data[randIdx[train_partition:]], target[randIdx[train_partition:]]



with tf.Session() as s:
	k = 1
	X = trainData

	predictY = knn_predict(s, X, testData, trainTarget, k=k)
	accuracy = s.run(compute_accuracy(testTarget, predictY))
	print accuracy

