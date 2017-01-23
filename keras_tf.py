import tensorflow as tf
from keras.metrics import categorical_accuracy as accuracy
from tensorflow.examples.tutorials.mnist import input_data
from keras.objectives import categorical_crossentropy
from keras.layers import Dense
from keras import backend as K

sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())

# this is placeholder for input features
img = tf.placeholder(tf.float32, shape=(None, 784))

x = Dense(128, activation='relu')(img)
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)

labels = tf.placeholder(tf.float32, shape=(None, 10))

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0], labels: batch[1]})


acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images, labels:
                                    mnist_data.test.labels})
