import os
import tensorflow as tf
import pickle

from preprocessing import Vocab


def fit_vocab(motif_data, num_words, dataset_dir, vocab_path):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=num_words,
                oov_token='<UNK>',
                filters='',
                lower=False,
                split='\t',
                char_level=True
    )

    vocab = Vocab(tokenizer)

    for data in motif_data:
        virusname = data['virus'].replace(' ', '_')

        for protein in data['proteins'].keys():
            dataset_path = os.path.join(os.path.join(
                    dataset_dir, virusname), f'{protein}.pickle')
            with open(dataset_path, 'rb') as f:
                x = pickle.load(f)

            vocab.fit(x)

    tokenizer = vocab.tokenizer
    with open(vocab_path, 'wb') as f:
        pickle.dump(tokenizer, f)
