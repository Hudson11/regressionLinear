import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#input features
x = tf.placeholder(tf.float32,[None, 1])
#output target
y_orig = tf.placeholder(tf.float32,[None, 1])

array = []

#weights and biases
w = tf.Variable(tf.zeros([1,1]), name='w')
b = tf.Variable(tf.zeros([1]), name='b')

y_pred = tf.matmul(x, w)+b
cost = tf.reduce_sum(tf.pow((y_orig - y_pred), 2))
learning_rate = 0.0001
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()
training_epochs = 100

valor = ''

sess = tf.Session()
sess.run(init)
for i in range(training_epochs):
    xs = np.array([[i]])
    ys = np.array([[(5*i+3)]])
    feed = {x:xs, y_orig:ys}
    sess.run(train_step, feed_dict=feed)
    #print('proxima interação %d' % i)
    #print('W: %f' % sess.run(w))
    #print('B: %f' % sess.run(b))
    print(sess.run(w).__getitem__(0))

    array.append(sess.run(w).__getitem__(0))

plt.plot([array])
plt.show()
