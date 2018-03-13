import tensorflow as tf
import numpy as np
import matplotlib .pyplot as plt

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


weight_decay = 0.0
loss = 0.5*tf.reduce_mean(tf.squared_difference(y, y_true)) + 0.5*weight_decay*tf.reduce_sum(tf.square(W))


learning_rate = 0.005
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


batch_size = 3500
epoch_ratio = int(train_size/batch_size)
iteration = 2000

loss_history = []

with tf.Session() as s:
    tf.global_variables_initializer().run()
    
    for step in xrange(iteration):
        offset = (step*batch_size) % num_label
        
        batch_data = trainData[offset:(offset + batch_size), :]
        batch_target = trainTarget[offset:(offset + batch_size)]
    
        train_step.run(feed_dict={x: batch_data, y_true: batch_target})

        if step%epoch_ratio==0:
            loss_history.append(s.run(loss, feed_dict={x: trainData, y_true: trainTarget}))

    
    print s.run(loss, feed_dict={x: trainData, y_true: trainTarget})


    plt.plot(range(1,len(loss_history)+1), loss_history, '-')
    plt.show()

