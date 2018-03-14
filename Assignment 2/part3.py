import tensorflow as tf
import numpy as np
import matplotlib .pyplot as plt
import pdb


# with np.load("notMNIST.npz") as data:
#     Data, Target = data ["images"], data["labels"]
#     np.random.seed(521)
#     randIndx = np.arange(len(Data))
#     np.random.shuffle(randIndx)
#     Data = np.reshape(Data, [-1, 28*28])
#     Data = Data[randIndx]/255.
#     Target = Target[randIndx]
#     trainData, trainTarget = Data[:15000], Target[:15000]
#     validData, validTarget = Data[15000:16000], Target[15000:16000]
#     testData, testTarget = Data[16000:], Target[16000:]


data = np.load('data.npy').astype(np.int32)
data = np.reshape(data, [-1, 32*32])
target = np.load('target.npy').astype(np.int64)
target = np.array(map(lambda x: x[0], target)) # Person

dataSize = len(data)

np.random.seed(521)
randIdx = np.arange(dataSize)
np.random.shuffle(randIdx)

train_partition = int(dataSize*0.8)
test_partition = int(dataSize*0.9)
data = data[randIdx]/255.
target = target[randIdx]
trainData, trainTarget = data[:train_partition], target[:train_partition]
testData, testTarget = data[train_partition:test_partition], target[train_partition:test_partition]
validData, validTarget = data[test_partition:], target[test_partition:]


num_label = 10
train_size, num_feature = np.shape(trainData)


x = tf.placeholder("float", shape=[None, num_feature])
y_true = tf.placeholder("float", shape=[None, num_label])

W = tf.Variable(tf.zeros([num_feature, num_label]))
b = tf.Variable(tf.zeros([num_label]))
y = tf.matmul(x, W) + b


fig_loss = plt.figure()
ax1 = fig_loss.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)

for learning_rate in [0.005, 0.001, 0.0001]:
    weight_decay = 0.01
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))\
            + 0.5*weight_decay*tf.reduce_sum(tf.square(W))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    batch_size = 300
    epoch_ratio = int(train_size/batch_size)
    iteration = 2000

    loss_history = []
    train_history = []
    test_history = []
    valid_history = []

    with tf.Session() as s:
        tf.global_variables_initializer().run()
        trainTarget_onehot = s.run(tf.one_hot(trainTarget, num_label))
        testTarget_onehot = s.run(tf.one_hot(testTarget, num_label))
        validTarget_onehot = s.run(tf.one_hot(validTarget, num_label))

        for step in xrange(iteration):
            offset = (step*batch_size) % num_label
            
            batch_data = trainData[offset:(offset + batch_size)]
            batch_target = trainTarget_onehot[offset:(offset + batch_size)]
        
            train_step.run(feed_dict={x: batch_data, y_true: batch_target})
        
            if step%epoch_ratio==0:
                randIndx = np.arange(len(trainData))
                np.random.shuffle(randIndx)
                trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]
                trainTarget_onehot = s.run(tf.one_hot(trainTarget, num_label))
                loss_history.append(s.run(loss, feed_dict={x: trainData, y_true: trainTarget_onehot}))
                train_history.append(accuracy.eval({x: trainData, y_true: trainTarget_onehot}))
                test_history.append(accuracy.eval({x: testData, y_true: testTarget_onehot}))
                valid_history.append(accuracy.eval({x: validData, y_true: validTarget_onehot}))


        ax1.plot(range(1,len(loss_history)+1), loss_history, '-', label='learning rate: %s' % learning_rate)
        ax2.plot(range(1,len(train_history)+1), train_history, '-', label='learning rate: %s' % learning_rate)
        ax3.plot(range(1,len(test_history)+1), test_history, '-', label='learning rate: %s' % learning_rate)
        ax4.plot(range(1,len(valid_history)+1), valid_history, '-', label='learning rate: %s' % learning_rate)

        print 'loss: %s' % s.run(loss, feed_dict={x: trainData, y_true: trainTarget_onehot})
        print 'train accuracy: %s' % accuracy.eval({x: trainData, y_true: trainTarget_onehot})
        print 'test accuracy: %s' % accuracy.eval({x: testData, y_true: testTarget_onehot})
        print 'validation accuracy: %s' % accuracy.eval({x: validData, y_true: validTarget_onehot})



ax1.set_xlabel('epoch')
ax2.set_xlabel('epoch')
ax3.set_xlabel('epoch')
ax4.set_xlabel('epoch')
ax1.set_ylabel('cross entropy loss')
ax2.set_ylabel('train accuracy')
ax3.set_ylabel('test accuracy')
ax4.set_ylabel('valid accuracy')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()