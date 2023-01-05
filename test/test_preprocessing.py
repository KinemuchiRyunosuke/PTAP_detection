import unittest
import numpy as np

from src.preprocessing import Vocab, underSampling


class TestVocab(unittest.TestCase):
    def test_separate_len1(self):
        seqs = ['ARNDCQEGHI',
                'LKMFPSTWYV']
        answer = [[0,1,2,3,4,5,6,7,8,9,10],
                  [0,11,12,13,14,15,16,17,18,19,20]]

        vocab = Vocab(separate_len=1, class_token=True)
        encoded_seqs = vocab.encode(seqs)

        self.assertEqual(encoded_seqs, answer)

        decoded_seqs = vocab.decode(encoded_seqs)
        self.assertEqual(decoded_seqs, seqs)

    def test_separate_len2(self):
        seqs = ['ARNDCQEGHI',
                'LKMFPSTWYV']

        vocab = Vocab(separate_len=2, class_token=True)
        encoded_seqs = vocab.encode(seqs)

        decoded_seqs = vocab.decode(encoded_seqs)
        self.assertEqual(decoded_seqs, seqs)

    def test_class_token_false(self):
        seqs = ['ARNDCQEGHI',
                'LKMFPSTWYV']
        answer = [[0,1,2,3,4,5,6,7,8,9],
                  [10,11,12,13,14,15,16,17,18,19]]

        vocab = Vocab(separate_len=1, class_token=False)
        encoded_seqs = vocab.encode(seqs)

        self.assertEqual(encoded_seqs, answer)

        decoded_seqs = vocab.decode(encoded_seqs)
        self.assertEqual(decoded_seqs, seqs)


class TestFuncs(unittest.TestCase):
    def test_under_sampling(self):
        x = np.arange(30).reshape(10, 3)
        y = (np.arange(10) >= 8).astype(int)
        x_resampled, y_resampled = underSampling(x, y, sampling_strategy=1)

        n_positive = (y_resampled == 1).sum()
        n_negative = (y_resampled == 0).sum()
        self.assertEqual(n_positive / n_negative, 1)

        x_positive = x_resampled[y_resampled == 1]
        x_positive_answer = np.array([[24, 25, 26],
                                      [27, 28, 29]])
        self.assertTrue((x_positive == x_positive_answer).all())


if __name__ == '__main__':
    unittest.main()