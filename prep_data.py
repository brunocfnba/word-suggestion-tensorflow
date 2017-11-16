from nltk import word_tokenize
import pickle


def create_word_list(source_ds):
    words_list = []
    num_lines = 0
    count = 0
    with open(source_ds, 'r', 20000, 'latin-1') as ds:
        with open('ds_to_list.pickle', 'wb', 10000) as ds_list:
            for line in ds:
                words = word_tokenize(line.lower())
                words_list += list(words)
                pickle.dump(words, ds_list)
                num_lines += 1

            word_count = set(words_list)
            word_list_final = {}
            for i in word_count:
                word_list_final[i] = count
                count += 1
            rev_word_dict = dict(zip(word_list_final.values(), word_list_final.keys()))

    list_size = len(word_list_final)

    print("Word dictionary size: {}".format(list_size))
    with open('word_dict.pickle', 'wb') as wd:
        pickle.dump(word_list_final, wd)

    with open('rev_word_dict.pickle', 'wb') as wd:
        pickle.dump(rev_word_dict, wd)

    print("Word dictionary generated and saved")
    return list_size, num_lines


with open('data_details.pkl', 'wb') as details:
    list_size, ds_size = create_word_list('question_data.txt')
    details_sizes = {'list': list_size, 'dataset': ds_size}
    pickle.dump(details_sizes, details)
