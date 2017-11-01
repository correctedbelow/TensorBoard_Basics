import sys
import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print('Python {}'.format(sys.version))
print('TensorFlow {}'.format(tf.__version__))

project_root = os.path.dirname(os.path.realpath(__file__))
tensorboard_logdir = os.path.join(project_root, 'tmp', 'tensorboard')
print('tensorboard --logdir=workaround:"' + tensorboard_logdir + '"') # See: https://github.com/tensorflow/tensorflow/issues/6313

tf.set_random_seed(195936478)

x = tf.placeholder(tf.float32, shape=(None, 2), name="x")
h = tf.layers.dense(x, units=2, activation=tf.sigmoid, use_bias=True, name="h")
y = tf.layers.dense(h, units=1, use_bias=True, name="y")
answers = tf.placeholder(tf.float32, shape=(None,1), name="answers")
with tf.name_scope("error"):
   mean_squared_error = tf.reduce_mean(tf.square(answers - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01) # Create an optimizer
train = optimizer.minimize(mean_squared_error)
session = tf.Session()

tf.summary.scalar("mean_squared_error", mean_squared_error)

summarizer = tf.summary.merge_all()

run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
summary_filename = os.path.join(tensorboard_logdir, run_name)
summary_writer = tf.summary.FileWriter(logdir=summary_filename, graph=session.graph)

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
   error, summary, _ = session.run([mean_squared_error, summarizer, train], {x: xor_inputs, answers: xor_outputs})
   summary_writer.add_summary(summary, i)

   if i % 250 == 0:
      print('mean_squared_error:', error)
prediction = session.run(y, {x: xor_inputs})
print(prediction)
