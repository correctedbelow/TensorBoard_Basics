import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print('Python {}'.format(sys.version))
print('TensorFlow {}'.format(tf.__version__))

tf.set_random_seed(195936478)

x = tf.placeholder(tf.float32, shape=(None, 2))
h = tf.layers.dense(x, units=2, activation=tf.sigmoid, use_bias=True)
y = tf.layers.dense(h, units=1, use_bias=True)
answers = tf.placeholder(tf.float32, shape=(None,1))
mean_squared_error = tf.reduce_mean(tf.square(answers - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01) # Create an optimizer
train = optimizer.minimize(mean_squared_error)
session = tf.Session()

session.run(tf.global_variables_initializer()) # REMEMBER: Always initialize your variables!

xor_inputs = [
       [0, 0],
       [0, 1],
       [1, 0],
       [1, 1]
       ]
xor_outputs = [
       [0],
       [1],
       [1],
       [0]
       ]

prediction = session.run(y, {x: xor_inputs})
print(prediction)


for i in range(2001):
   error, _ = session.run([mean_squared_error, train], {x: xor_inputs, answers: xor_outputs})
   if i % 250 == 0:
      print('mean_squared_error:', error)
prediction = session.run(y, {x: xor_inputs})
print(prediction)
