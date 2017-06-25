# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import collections

f = open('./poetry/poetry.txt')
n = 0
poetries = []
for line in f:
    line = line.decode('UTF8').strip('\n')
    poetries.append(line)

all_words = []
for p in poetries:
    all_words += [word for word in p]
counter = collections.Counter(all_words)

counter = sorted(counter.items(), key = lambda x:-x[1])

idx_to_words, _ = zip(*counter)
idx_to_words = idx_to_words[:len(idx_to_words)] + (' '.decode('UTF8'),)

tf.reset_default_graph()
batch_size = 8
vocab_size = len(idx_to_words)
hidden_size = 16
num_steps = 10
#lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
#cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2)
cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(hidden_size) for _ in range(2)])

#input_data = tf.placeholder(tf.int32, [None, None])#第一个维度是batch_size,第二个维度是num_steps
#targets = tf.placeholder(tf.int32, [None, None])#跟上面一致

#embedding = tf.get_variable(name='embedding', shape=[len(idx_to_words), hidden_size])

#inputs = tf.nn.embedding_lookup(embedding, input_data)
#outputs = []

#initial_state = cell.zero_state(batch_size, tf.float32)

#state = initial_stat

#with tf.variable_scope('RNN'):
#    for time_step in range(dataset.sample_len):
#        if time_step > 0: tf.get_variable_scope().reuse_variables()
#        cell_output, state = cell(inputs[:,time_step,:], state)
#        outputs.append(cell_output)
inputs = tf.placeholder(tf.float32,[None,None,16])
        
outputs, final_state = tf.nn.dynamic_rnn(cell, inputs ,dtype=tf.float32)
#output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
#output = tf.reshape(outputs, [-1, hidden_size])


#weight = tf.get_variable('weight', [hidden_size, vocab_size])
#bias = tf.get_variable('bias', [vocab_size])
#logits = tf.matmul(output, weight) + bias

#loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(targets, [-1])],
#                                                          [tf.ones_like(targets, dtype = tf.float32)])
#cost = tf.reduce_sum(loss) / batch_size

#final_state = state

trainable_variables = tf.trainable_variables()
#grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainable_variables), clip_norm=5)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
#train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
print trainable_variables, '-------------------'

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './model/model.ckpt')