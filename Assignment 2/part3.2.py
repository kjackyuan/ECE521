import tensorflow as tf
import numpy as np
import matplotlib .pyplot as plt


data = np.load('data.npy').astype(np.int32)
data = np.reshape(data, [-1, 32*32])
target = np.load('target.npy').astype(np.int64)
target = np.array(map(lambda x: x[0], target)) # Person

dataSize = len(data)

np.random.seed(521)
randIdx = np.arange(dataSize)
np.random.shuffle(randIdx)

train_partition = int(dataSize*0.8)
data = data[randIdx]/255.
target = target[randIdx]
trainData, trainTarget = data[:train_partition], target[:train_partition]
testData, testTarget = data[train_partition:], target[train_partition:]


num_label = 10
train_size, num_feature = np.shape(trainData)


x = tf.placeholder("float", shape=[None, num_feature])
y_true = tf.placeholder("float", shape=[None, num_label])

W = tf.Variable(tf.zeros([num_feature, num_label]))
b = tf.Variable(tf.zeros([num_label]))
y = tf.matmul(x, W) + b


weight_decay = 0.01
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))\
        + 0.5*weight_decay*tf.reduce_sum(tf.square(W))


learning_rate = 0.005
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_size = 500
epoch_ratio = int(train_size/batch_size)
iteration = 2000

loss_history = []
accuracy_history = []

with tf.Session() as s:
    tf.global_variables_initializer().run()
    trainTarget_onehot = s.run(tf.one_hot(trainTarget, num_label))
    testTarget_onehot = s.run(tf.one_hot(testTarget, num_label))
    
    for step in xrange(iteration):
        offset = (step*batch_size) % train_size
        
        batch_data = trainData[offset:(offset + batch_size), :]
        batch_target = trainTarget_onehot[offset:(offset + batch_size)]
    
        train_step.run(feed_dict={x: batch_data, y_true: batch_target})
    
        if step%epoch_ratio==0:
            loss_history.append(s.run(loss, feed_dict={x: trainData, y_true: trainTarget_onehot}))
            accuracy_history.append(accuracy.eval({x: testData, y_true: testTarget_onehot}))


    print accuracy.eval({x: testData, y_true: testTarget_onehot})
    epoch_range = range(1,len(loss_history)+1)
    plt.plot(epoch_range, loss_history, '-', epoch_range, accuracy_history, 'ro')
    plt.show()
