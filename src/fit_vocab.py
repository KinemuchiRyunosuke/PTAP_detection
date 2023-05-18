import os
import tensorflow as tf
import pickle

from preprocessing import Vocab


def fit_vocab(motif_data, num_words, dataset_dir):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=num_words,
                oov_token='X',
                filters='',
                lower=False,
                split=' ',
                char_level=False
    )

    vocab = Vocab(tokenizer)

    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        dataset_path = os.path.join(dataset_dir, f'{virus}.pickle')

        with open(dataset_path, 'rb') as f:
            x = pickle.load(f)

        vocab.fit(x)

    return vocab