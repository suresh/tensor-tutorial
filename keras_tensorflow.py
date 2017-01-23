import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data
from keras.metrics import categorical_accuracy as accuracy

sess = tf.Session()
K.set_session(sess)

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

x = Dense(128, activation='relu')(img)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
sess.run(tf.global_variables_initializer())

with sess.as_default():
    for i in range(500):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels:
                                  batch[1],
                                  K.learning_phase(): 1})


acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images, labels:
                                    mnist_data.test.labels,
                                    K.learning_phase(): 0})
