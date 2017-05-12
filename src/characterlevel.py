import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import process


LOG_DIR = '../logs'

char_dictionary, maxlen, dictionary, data = process.generate_data('../data/rhymes.txt', char_level=True)
reverse_dictionary = {index: word for word, index in dictionary.items()}
process.write_vocab_file(dictionary, os.path.join(LOG_DIR, 'metadata.tsv'))

np.random.shuffle(data)
train_split = .8
train_split_num = int(len(data) * train_split)
train_data = data[:train_split_num]
test_data = data[train_split_num:]

vocab_size = len(dictionary)
embedding_size = 64
lstm_size = 64
concat_size = embedding_size + lstm_size * 2
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
    tf.truncated_normal([vocab_size, concat_size],
                        stddev=1.0 / math.sqrt(concat_size)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

# placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size, maxlen + 2])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# split input for word embeddings, word length, and character model
word_inputs = train_inputs[:, 0]
lengths = train_inputs[:, 1]
character_inputs = tf.one_hot(train_inputs[:, 2:], 26)

word_embed = tf.nn.embedding_lookup(embedding_var, word_inputs)

# character model
rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=tf.contrib.rnn.LSTMCell(lstm_size),
    cell_bw=tf.contrib.rnn.LSTMCell(lstm_size),
    inputs=character_inputs, sequence_length=lengths, dtype=tf.float32)
fw_output, bw_output = rnn_output
rnn_concat = tf.concat([fw_output, bw_output], 2)

# select the last output based on the length
indices = tf.range(batch_size) * maxlen + (lengths - 1)
flat = tf.reshape(rnn_concat, [-1, lstm_size * 2])
last = tf.gather(flat, indices)

# concatenate the word embeddings and character model output
final_embed = tf.concat([word_embed, last], 1)

# computer the nce loss
loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=train_labels,
                   inputs=final_embed,
                   num_sampled=num_sampled,
                   num_classes=vocab_size))

# compute final output for testing purposes
final = tf.matmul(final_embed, tf.transpose(nce_weights)) + nce_biases
softmax = tf.nn.softmax(final)

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
        # inp = session.run([word_inputs], feed_dict=feed_dict)
        # print(inp)
        _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)

        if i % 10000 == 0:
            print(str(i) + ': ' + str(cur_loss))
            e, s = session.run([word_embed, softmax],
                               feed_dict={train_inputs: [char_dictionary['gun']] * batch_size})
            words = np.argsort(s[0])[::-1]
            print([s[0][index] for index in words[:10]])
            print([reverse_dictionary[index] for index in words[:10]])
            print()

        if i % 25000 == 0:
            test_batch_generator = process.batch_generator(test_data, batch_size)
            for i in range(10):
                inputs, labels = next(test_batch_generator)
                feed_dict = {train_inputs: inputs, train_labels: labels}
                [cur_loss] = session.run([loss], feed_dict=feed_dict)
                print(cur_loss)


    print(str(i) + ': ' + str(cur_loss))
    print('\nFinished training')

    saver.save(session, os.path.join(LOG_DIR, 'model.ckpt'))

# with tf.Session() as session:
#     saver.restore(session, os.path.join(LOG_DIR, 'model.ckpt'))

#     e, s = session.run([word_embed, softmax],
#                        feed_dict={train_inputs: [char_dictionary['hook']] * batch_size})
#     words = np.argsort(s[0])[::-1]
#     print([s[0][index] for index in words[:10]])
#     print([reverse_dictionary[index] for index in words[:10]])
