import tensorflow as tf
import numpy as np
import pdb
import scipy.misc

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

trainData = np.concatenate([trainData[trainTarget == 0], trainData[trainTarget == 1]])
trainTarget = np.concatenate([trainTarget[trainTarget == 0], trainTarget[trainTarget == 1]])

testData = np.concatenate([testData[testTarget == 0], testData[testTarget == 1]])
testTarget = np.concatenate([testTarget[testTarget == 0], testTarget[testTarget == 1]])


n_classes = 2
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

keep_rate = 0.6

layer1 = tf.nn.relu(build_layer(x, 100))

y = build_layer(layer1, n_classes) #output layer

all_weight = tf.concat(layer_weight, 0)

weight_decay = 0.0003
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))# + 0.5*weight_decay*tf.reduce_sum(tf.square(all_weight))

#learning rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

batch_size = 500
iteration = 10*batch_size

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def normalize(mat):
    mat_min = np.min(mat)
    mat = mat - mat_min
    mat_max = np.max(mat)
    mat = mat/mat_max
    mat = mat*255.0
    return mat

def visualize(perc):
    a = s.run(tf.reshape(tf.transpose(tf.reshape(layer_weight[0], [784, 100])), [100, 28, 28]))
    final_img = None

    for col in range(10):
        col_img = normalize(a[col*10])
        for row in range(1, 10):
            col_img = np.concatenate([col_img, np.zeros((1, 28)), normalize(a[col*10+row])])
        if final_img is None:
            final_img = np.transpose(col_img)
        else:
            col_size = len(col_img)
            col_img = np.transpose(col_img)
            final_img = np.concatenate([final_img, np.zeros((1, col_size)), col_img])

    scipy.misc.imsave('%s.jpg'%perc, final_img)


with tf.Session() as s:
    tf.global_variables_initializer().run()
    
    for step in xrange(iteration):
        offset = (step*batch_size) % train_size
        
        batch_data = trainData[offset:(offset + batch_size)]
        batch_target = trainTarget_oh[offset:(offset + batch_size)]
    
        train_step.run(feed_dict={x: batch_data, y_true: batch_target})
    
        if step%(batch_size)==0:
            randIndx = np.arange(len(trainData))
            np.random.shuffle(randIndx)
            trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]
            trainTarget_oh = s.run(tf.one_hot(trainTarget, n_classes))
        
        if step%(batch_size) == 0:
            print s.run(loss, feed_dict={x: trainData, y_true: trainTarget_oh})
            print accuracy.eval({x: testData, y_true: testTarget_oh})
            print ''

        if step%batch_size == 0:
            visualize(step/batch_size)