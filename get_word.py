import numpy as np
from nltk import word_tokenize
import tensorflow as tf
import pickle
from lstm_sentence_nn import lstm_rnn, training


interval_size = 3


def get_top(topn, the_list):
    ordered_list = []
    for i in range(len(the_list)):
        if i == 0:
            ordered_list.append(the_list[i])
        else:
            if the_list[i] < ordered_list[0]:
                ordered_list.insert(0, the_list[i])
            else:
                for j in range(len(ordered_list)):
                    if j == (len(ordered_list)-1):
                        ordered_list.append(the_list[i])
                    else:
                        if ordered_list[j] < the_list[i] < ordered_list[j+1]:
                            ordered_list.insert(j+1, the_list[i])
    biggest_values = ordered_list[-topn:]
    index_list = []
    for i in biggest_values:
        index_list.append(the_list.index(i))
    return index_list


def get_word():
        with open('word_dict.pickle', 'rb') as wd:
            word_dict = pickle.load(wd)
        with open('rev_word_dict.pickle', 'rb') as rwd:
            rev_word_dict = pickle.load(rwd)
            in_placeholder = tf.placeholder('float', [None, interval_size, 1])
            rnn_output = lstm_rnn(in_placeholder)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, 'model.ckpt')
                phrase_check = 'invalid'
                while phrase_check == 'invalid':
                    prompt = 'Type at least {} words: '.format(interval_size)
                    phrase = input(prompt)
                    phrase = phrase.strip()
                    words = word_tokenize(phrase.lower())
                    if len(words) < interval_size:
                        print('We need at least {} words!'.format(interval_size))
                    else:
                        words = words[-interval_size:]
                        phrase_check = 'valid'
                next_word = '-'
                word_dict_size = len(word_dict)
                phrase_to_num = []
                for i in words:
                    if i in word_dict.keys():
                        phrase_to_num.append(word_dict[i])
                    else:
                        phrase_to_num.append(word_dict_size)
                        word_dict_size += 1

                while next_word != '4':
                    phrase_reshape = np.reshape(np.array(phrase_to_num), [-1, interval_size, 1])
                    rnn_predictions = sess.run(rnn_output, feed_dict={in_placeholder: phrase_reshape})
                    answers = get_top(3, list(rnn_predictions[0]))
                    print('Suggestions are:')
                    num = 1
                    for j in answers:
                        print('{}: {}'.format(num, rev_word_dict[j]))
                        num += 1
                    print('{}: Finish phrase\n'.format(num))
                    prompt = 'Select the number or type a word: '
                    next_word = input(prompt)

                    if next_word in ['1', '2', '3']:
                        phrase = phrase + ' ' + rev_word_dict[answers[int(next_word)-1]]
                        print('Your phrase so far: {}'.format(phrase))
                        phrase_to_num = phrase_to_num[1:]
                        phrase_to_num.append(answers[int(next_word)-1])
                    elif next_word == '4':
                        print('Final phrase: {}'.format(phrase))
                    else:
                        phrase = phrase + ' ' + next_word
                        print('Your phrase so far: {}'.format(phrase))
                        phrase_to_num = phrase_to_num[1:]
                        if next_word in word_dict.keys():
                            phrase_to_num.append(word_dict[next_word])
                        else:
                            phrase_to_num.append(word_dict_size)
                            word_dict_size += 1

# Uncomment the 2 following rows to train the LSTM RNN
# x = tf.placeholder('float', [None, interval_size, 1])
# training(x)


get_word()
