# These functions are responsible for generating the slogans from the trained LSTM model.
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, model_from_json
import pandas as pd
import warnings
import random

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# This function is for generate a single slogan text given inputs requirements
def generate_tagline(seed_text, next_words, loaded_model, max_len, t):
    for _ in range(next_words):
        token_list = t.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
        predicted = loaded_model.predict_classes(token_list, verbose=0)
        output_word = ''
        for word, index in t.word_index.items():
            if index == predicted:
                output_word = word
                break
        if output_word == '<end>':
            break
        seed_text = seed_text + " " + output_word
    return seed_text.title()


# This function is for generating a list of slogans appropriate for the coffee context of this project.
def get_taglines():
    directory = 'Sequence_model'
    coffee_df = pd.read_csv('{}/coffee.csv'.format(directory), encoding="latin-1")
    all_taglines = list(coffee_df.values)
    corpus = [str(x[0]) + ' <end>' for x in all_taglines]
    tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', lower=True, split=' ',
                          char_level=False, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(corpus)
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(seq) for seq in input_sequences])

    # load json and create model
    json_file = open('{}/LSTM_model.json'.format(directory), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("{}/LSTM_model.h5".format(directory))
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')

    slogans = []
    list_of_words = []
    for word, index in tokenizer.word_index.items():
        list_of_words.append(word)
    for _ in range(100):
        chosen_word = random.sample(list_of_words, 1)[0]
        output = generate_tagline(chosen_word, 20, loaded_model, max_sequence_len, tokenizer)
        # Only filter outputs have the coffee-related words in it.
        if 'Coffee' or 'coffee' in output:
            slogans.append(output)
    return slogans
