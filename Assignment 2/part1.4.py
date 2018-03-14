import tensorflow as tf
import numpy as np
import matplotlib .pyplot as plt
import pdb

with np.load("notMNIST.npz") as data :
    Data, Target = data ["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = np.reshape(Data, [-1, 28*28])
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    Data = np.array(map(lambda item: np.append(item, [1.0]), Data))
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]


with tf.Session() as sess:
    x_train = trainData
    y_train = trainTarget
    x_test = testData
    y_test = testTarget
    x_valid = validData
    y_valid = validTarget

    x_t = tf.transpose(x_train)
    W = sess.run(tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(x_t, x_train)), x_t), tf.cast(y_train, tf.float64)))


    pred_test = sess.run(tf.matmul(x_test, W))
    delta_test = tf.abs(pred_test - y_test)
    correct_prediction_test = tf.cast(tf.less(delta_test, 0.5), tf.int32)
    accuracy_test = sess.run(tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32)))


    pred_train = tf.matmul(x_train, W)
    delta_train = tf.squared_difference(pred_train, y_train)
    correct_prediction_train = tf.cast(tf.less(delta_train, 0.5), tf.int32)
    accuracy_train = sess.run(tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32)))
    mse = sess.run(0.5*tf.reduce_mean(delta_train))


    pred_valid = sess.run(tf.matmul(x_test, W))
    delta_valid = tf.abs(pred_valid - y_test)
    correct_prediction_valid = tf.cast(tf.less(delta_valid, 0.5), tf.int32)
    accuracy_valid = sess.run(tf.reduce_mean(tf.cast(correct_prediction_valid, tf.float32)))


    print 'MSE: %s' % mse
    print 'training accuracy: %s' % accuracy_train
    print 'validation accuracy: %s' % accuracy_test
    print 'test accuracy: %s' % accuracy_valid

