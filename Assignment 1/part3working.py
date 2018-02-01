import tensorflow as tf
import numpy as np

def sqr_distance(X, Z):
	X = tf.expand_dims(X, 1)
	Z = tf.expand_dims(Z, 0)
	T = tf.square(X-Z)
	T = tf.reduce_sum(T, axis=3)
	T = tf.reduce_sum(T, axis=2)
	return T

def knn_predict(X, pt, Y, k=1):
	dist = sqr_distance(X, pt)
	_, idx = tf.nn.top_k(-1*tf.transpose(dist), k=k)
	gather = tf.gather(Y, idx)

	counts = [
		tf.count_nonzero(gather-0, axis=1),
		tf.count_nonzero(gather-1, axis=1),
		tf.count_nonzero(gather-2, axis=1),
		tf.count_nonzero(gather-3, axis=1),
		tf.count_nonzero(gather-4, axis=1),
		tf.count_nonzero(gather-5, axis=1)
	]
	prediction = tf.argmin(counts, axis=0)
	return prediction

def compute_accuracy(trueY, predictY):
	eq = tf.equal(trueY, predictY)
	accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))
	return accuracy


data = np.load('data.npy').astype(np.int32)
target = np.load('target.npy').astype(np.int64)
target = np.array(map(lambda x: x[1], target)) # Gender
# target = np.array(map(lambda x: x[0], target)) # Person
dataSize = len(data)

np.random.seed(521)
randIdx = np.arange(dataSize)
np.random.shuffle(randIdx)

train_partition = int(dataSize*0.8)
trainData, trainTarget = data[randIdx[:train_partition]], target[randIdx[:train_partition]]
testData, testTarget = data[randIdx[train_partition:]], target[randIdx[train_partition:]]



with tf.Session() as s:
	X = trainData

	for k in [1,5,10,25,50,100,200]:
		accuracy = s.run(compute_accuracy(testTarget, knn_predict(X, testData, trainTarget, k=k)))
		print "K=%s, Accuracy=%s" % (k, accuracy)
		print "\n"

