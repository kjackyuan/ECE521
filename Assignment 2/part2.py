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

    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]


train_size, num_feature = np.shape(trainData)
num_label = 1


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

for learning_rate in [0.001]:
    weight_decay = 0.0

    for train_model in ['Linear', 'Logistic']:
        if train_model == 'Logistic':
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y)) + 0.5*weight_decay*tf.reduce_sum(tf.square(W))
        else:
            loss = 0.5*tf.reduce_mean(tf.squared_difference(y, y_true)) + 0.5*weight_decay*tf.reduce_sum(tf.square(W))

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        if train_model == 'Logistic':
            delta = tf.abs(y_true - tf.sigmoid(y))
        else:
            delta = tf.abs(y_true - y)

        correct_prediction = tf.cast(tf.less(delta, 0.5), tf.int32)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        batch_size = 500
        epoch_ratio = int(train_size/batch_size)
        iteration = 5000

        loss_history = []
        train_history = []
        test_history = []
        valid_history = []

        with tf.Session() as s:
            tf.global_variables_initializer().run()
            
            for step in xrange(iteration):
                offset = (step*batch_size) % num_label
                
                batch_data = trainData[offset:(offset + batch_size)]
                batch_target = trainTarget[offset:(offset + batch_size)]
            
                train_step.run(feed_dict={x: batch_data, y_true: batch_target})

                if step%epoch_ratio==0:
                    randIndx = np.arange(len(trainData))
                    np.random.shuffle(randIndx)
                    trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]
                    loss_history.append(s.run(loss, feed_dict={x: trainData, y_true: trainTarget}))

                    train_history.append(accuracy.eval({x: trainData, y_true: trainTarget}))
                    test_history.append(accuracy.eval({x: testData, y_true: testTarget}))
                    valid_history.append(accuracy.eval({x: validData, y_true: validTarget}))

            ax1.plot(range(1,len(loss_history)+1), loss_history, '-', label='model: %s' % train_model)
            ax2.plot(range(1,len(train_history)+1), train_history, '-', label='model: %s' % train_model)
            ax3.plot(range(1,len(test_history)+1), test_history, '-', label='model: %s' % train_model)
            ax4.plot(range(1,len(valid_history)+1), valid_history, '-', label='model: %s' % train_model)

            print 'loss: %s' % s.run(loss, feed_dict={x: trainData, y_true: trainTarget})
            print 'train accuracy: %s' % accuracy.eval({x: trainData, y_true: trainTarget})
            print 'test accuracy: %s' % accuracy.eval({x: testData, y_true: testTarget})
            print 'validation accuracy: %s' % accuracy.eval({x: validData, y_true: validTarget})


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