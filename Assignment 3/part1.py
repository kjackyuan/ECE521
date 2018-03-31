import tensorflow as tf
import numpy as np

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    Data = np.reshape(Data, [-1, 28*28])
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]


n_classes = 10
num_feature = 28*28
train_size = len(trainData)

with tf.Session() as s:
    testTarget_oh = s.run(tf.one_hot(testTarget, n_classes))
    trainTarget_oh = s.run(tf.one_hot(trainTarget, n_classes))
    validTarget_oh = s.run(tf.one_hot(validTarget, n_classes))

layer_weight = []

def build_layer(prev_layer, num_node):
    prev_num_node = prev_layer.shape[1].value
    xavier_stddev = tf.sqrt(3.0/(prev_num_node + num_node))
    W = tf.Variable(tf.random_normal(shape=[prev_num_node, num_node], stddev=xavier_stddev))
    b = tf.Variable(tf.zeros([num_node]))

    layer_weight.append(tf.reshape(W,[-1]))
    return tf.add(tf.matmul(prev_layer, W), b)

x = tf.placeholder("float", shape=[None, num_feature])
y_true = tf.placeholder("float", shape=[None, n_classes])

keep_rate = 0.7

layer1 = tf.nn.dropout(tf.nn.relu(build_layer(x, 500)), keep_rate)

layer2 = tf.nn.dropout(tf.nn.relu(build_layer(layer1, 500)), keep_rate)

layer3 = tf.nn.dropout(tf.nn.relu(build_layer(layer2, 500)), keep_rate)

assert False

y = build_layer(layer3, n_classes) #output layer

all_weight = tf.concat(layer_weight, 0)

weight_decay = 0.0003
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y)) + 0.5*weight_decay*tf.reduce_sum(tf.square(all_weight))

#learning rate = 0.001
train_step = tf.train.AdamOptimizer().minimize(loss)

batch_size = 500
epoch_ratio = int(train_size/batch_size)
iteration = 15000

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as s:
    tf.global_variables_initializer().run()
    
    for step in xrange(iteration):
        offset = (step*batch_size) % train_size
        
        batch_data = trainData[offset:(offset + batch_size)]
        batch_target = trainTarget_oh[offset:(offset + batch_size)]
    
        train_step.run(feed_dict={x: batch_data, y_true: batch_target})
    
        if step%(epoch_ratio)==0:
            randIndx = np.arange(len(trainData))
            np.random.shuffle(randIndx)
            trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]
            trainTarget_oh = s.run(tf.one_hot(trainTarget, n_classes))
        
        if step%(epoch_ratio) == 0:
            print s.run(loss, feed_dict={x: trainData, y_true: trainTarget_oh})
            print accuracy.eval({x: testData, y_true: testTarget_oh})
            print ''