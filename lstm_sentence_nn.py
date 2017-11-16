import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle


def load_details():
    with open('data_details.pkl', 'rb') as details:
        det = pickle.load(details)
        return det


files_details = load_details()

epochs = 200
interval_size = 3
num_neurons = 600

x = tf.placeholder('float', [None, interval_size, 1])
y = tf.placeholder('float', [None, files_details['list']])


def lstm_rnn(tf_placeholder):

    output_weight = tf.Variable(tf.random_normal([num_neurons, files_details['list']]))
    output_biases = tf.Variable(tf.random_normal([files_details['list']]))

    tf_placeholder = tf.reshape(tf_placeholder, [-1, interval_size])

    tf_placeholder = tf.split(tf_placeholder, interval_size, 1)

    rnn_layers = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_neurons), rnn.BasicLSTMCell(num_neurons)])

    outputs, states = rnn.static_rnn(rnn_layers, tf_placeholder, dtype=tf.float32)

    return tf.matmul(outputs[-1], output_weight) + output_biases


def training(in_placeholder):
    rnn_output = lstm_rnn(in_placeholder)
    saver = tf.train.Saver()

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn_output, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)

    correct = tf.equal(tf.argmax(rnn_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        step = 0
        start = 0
        end = interval_size
        acc_total = 0
        loss_total = 0
        counter = 0

        with open('word_dict.pickle', 'rb') as wd:
            word_dict = pickle.load(wd)
            while step < epochs:
                with open('ds_to_list.pickle', 'rb', 20000) as ds:
                    for _ in range(files_details['dataset']):
                        line_list = pickle.load(ds)
                        # print('line: {}'.format(line_list))
                        line_size = len(line_list)
                        while end < line_size:
                            sequence = [[word_dict[i]] for i in line_list[start:end]]
                            sequence = np.array(sequence)
                            sequence = np.reshape(sequence, [-1, interval_size, 1])
                            label = line_list[end]
                            label_hot_vector = np.zeros([files_details['list']])
                            label_hot_vector[word_dict[label]] = 1.0
                            label_hot_vector = np.reshape(label_hot_vector, [1, -1])

                            start += 1
                            end += 1

                            _, acc, loss, rnn_predictions = session.run([optimizer, accuracy, cost, rnn_output],
                                                                        feed_dict={in_placeholder: sequence,
                                                                        y: label_hot_vector})
                            counter += 1
                            acc_total += acc
                            loss_total += loss

                print('{}. Loss: {:.4f} and Accuracy: {:.2f}%'.format(step+1, loss_total / counter,
                                                                      (100 * acc_total) / counter))
                acc_total = 0
                loss_total = 0
                counter = 0

                start = 0
                end = interval_size

                step += 1
            saver.save(session, 'model.ckpt')
            print('Training completed')


# training(x)
