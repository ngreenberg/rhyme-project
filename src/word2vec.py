import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import process


LOG_DIR = '../logs'

dictionary, data = process.generate_data('../data/rhymes.txt')
reverse_dictionary = {index: word for word, index in dictionary.items()}
process.write_vocab_file(dictionary, os.path.join(LOG_DIR, 'metadata.tsv'))

np.random.shuffle(data)
train_split = .8
train_split_num = int(len(data) * train_split)
train_data = data[:train_split_num]
test_data = data[train_split_num:]

vocab_size = len(dictionary)
embedding_size = 64
batch_size = 64
num_sampled = 10

# input embeddings
embedding_var = tf.Variable(
    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

# add metadata for tensorboard
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
summary_writer = tf.summary.FileWriter(LOG_DIR)
projector.visualize_embeddings(summary_writer, config)

# output weights
nce_weights = tf.Variable(
    tf.truncated_normal([vocab_size, embedding_size],
                        stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

# placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embed = tf.nn.embedding_lookup(embedding_var, train_inputs)

# computer the nce loss
loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=train_labels,
                   inputs=embed,
                   num_sampled=num_sampled,
                   num_classes=vocab_size))

# compute output for testing purposes
output = tf.nn.softmax(tf.matmul(embed, tf.transpose(nce_weights)) + nce_biases)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
batch_generator = process.batch_generator(train_data, batch_size)
with tf.Session() as session:
    init.run()

    print('\nTraining...\n')
    for i in range(2000000):
        inputs, labels = next(batch_generator)
        feed_dict = {train_inputs: inputs, train_labels: labels}
        _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)

        if i % 10000 == 0:
            print(str(i) + ': ' + str(cur_loss))
            e, o = session.run([embed, output],
                               feed_dict={train_inputs: [dictionary['one']] * batch_size})
            words = np.argsort(o[0])[::-1]
            print([o[0][index] for index in words[:10]])
            print([reverse_dictionary[index] for index in words[:10]])
            print()
    print(str(i) + ': ' + str(cur_loss))
    print('\nFinished training')

    batch_generator = process.batch_generator(test_data, batch_size)
    for i in range(10):
        inputs, labels = next(batch_generator)
        feed_dict = {train_inputs: inputs, train_labels: labels}
        [cur_loss] = session.run([loss], feed_dict=feed_dict)
        print(cur_loss)

    saver.save(session, os.path.join(LOG_DIR, 'model.ckpt'))
