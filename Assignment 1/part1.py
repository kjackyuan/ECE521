import tensorflow as tf
import numpy as np

def sqr_distance(X, Z):
	X = tf.expand_dims(X, 1)
	Z = tf.expand_dims(Z, 0)
	T = tf.square(X - Z)
	T = tf.reduce_sum(T, axis=2)
	return T

with tf.Session() as ss:
	a = tf.constant([[0,0,0], [1,1,1], [2,2,2], [3,3,3]])
	b = tf.constant([[0,0,0], [1,1,1]])
	dist = ss.run(sqr_distance(a, b))

	print dist