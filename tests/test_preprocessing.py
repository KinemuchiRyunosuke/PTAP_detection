
import unittest
import numpy as np
import tensorflow as tf

from src.preprocessing import Vocab, under_sampling, add_class_token


class TestVocab(unittest.TestCase):
    def test_encode_decode(self):
        texts = ['ABCDEFG',
                 'HIJKLMNOPQR',
                 'STUVWXYZ',
                 'HELLO',
                 'WORLD']

        tokenizer = tf.keras.preprocessing.text.Tokenizer(
                    num_words=28,
                    oov_token='<UNK>',
                    filters='',
                    lower=False,
                    split='\t',
                    char_level=True
        )

        vocab = Vocab(tokenizer)
        vocab = vocab.fit(texts)

        sequences = vocab.encode(texts)
        decoded_texts = vocab.decode(sequences)
        self.assertEqual(decoded_texts, texts)

        # class_token を付加した場合
        sequences = vocab.encode(texts)
        sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, padding='post', value=0)
        sequences = add_class_token(sequences)
        decoded_texts = vocab.decode(sequences, class_token=True)
        self.assertEqual(decoded_texts, texts)

        # encodeの入力がndarrayの場合
        ndarray_texts = np.array(texts)
        sequences = vocab.encode(ndarray_texts)
        decoded_texts = vocab.decode(sequences)
        self.assertEqual(decoded_texts, texts)


class TestFuncs(unittest.TestCase):
    def test_add_class_token(self):
        sequences = [[23, 4, 2, 8],
                     [1,  8, 5, 2],
                     [9, 13, 4, 1]]
        correct_answer = np.array([[0, 23,  4, 2, 8],
                                   [0,  1,  8, 5, 2],
                                   [0,  9, 13, 4, 1]])
        answer = add_class_token(sequences)
        self.assertTrue((answer == correct_answer).all())

    def test_under_sampling(self):
        x = np.arange(30).reshape(10, 3)
        y = (np.arange(10) >= 8).astype(int)
        x_resampled, y_resampled = under_sampling(x, y, sampling_strategy=1)

        n_positive = (y_resampled == 1).sum()
        n_negative = (y_resampled == 0).sum()
        self.assertEqual(n_positive / n_negative, 1)

        x_positive = x_resampled[y_resampled == 1]
        x_positive_answer = np.array([[24, 25, 26],
                                      [27, 28, 29]])
        self.assertTrue((x_positive == x_positive_answer).all())


if __name__ == '__main__':
    unittest.main()