from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pandas as pd


# 导入数据集
df = pd.read_csv('G:/AutoEncode/new/数据/4X油膜振荡train.csv', engine='python')


# 参数
learning_rate = 0.01  # 学习率
training_epochs = 10000  # 训练的周期
batch_size = 125  # 每一批次训练的大小
REGULARIZATION_RATE = 0.01

# 神经网络的参数
n_input = 160  # 输入
n_hidden_1 = 60  # 隐层1的神经元个数
# n_hidden_2 = 40  # 隐层2神经元个数

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    # 'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    # 'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    # 'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    # 'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_1


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    # layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.add(tf.matmul(x, weights['decoder_h2']), biases['decoder_b2'])
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# 正则化
# regularizer = tf.contrib.layers.l1_regularizer(REGULARIZATION_RATE)
# regularization = regularizer(encoder_op)


# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
# cost = -tf.reduce_mean(y_true*tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))
# cost = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y_pred),1))
# cost = tf.reduce_mean(y_pred-y_true)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

# 计算批次
total_batch = int(df.shape[0] / batch_size)
# Training cycle
# temp = np.zeros((1, 1000), dtype=float)
for epoch in range(training_epochs):
    # Loop over all batches
    k = 0
    for i in range(total_batch):
        batch_xs = df[k:k+batch_size]
        k = k+batch_size
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # c = sess.run(cost, feed_dict={X: batch_xs})
    # Display logs per epoch step
    learning_rate = learning_rate * 0.99
    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    # temp[0][epoch] = c
# pd_temp = pd.DataFrame(temp)
# pd_temp.to_excel('1.xlsx')
print("Optimization Finished!")

saver = tf.train.Saver()
saver.save(sess, "./model/model1/model1.ckpt")
sess.close()
