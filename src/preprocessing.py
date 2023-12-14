import numpy as np
import itertools

import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler

class Vocab:
    def __init__(self, separate_len):
        """
        Args:
            separate_len(int): n連続アミノ酸頻度でIDに変換する

        """
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

        vocab_list = itertools.product(amino_acids, repeat=separate_len)
        vocab_list = [''.join(vocab) for vocab in vocab_list]

        encode_table = {'<PAD>': 0, '<CLS>': 1, 'X': 2}     # <PAD>: padding token
        decode_table = {0: '<PAD>', 1: '<CLS>', 2: 'X'}     # <CLS>: class token
        for i, amino_acid in enumerate(vocab_list):     # X: 未知のアミノ酸，または
            encode_table[amino_acid] = i + 3            # 上記アミノ酸以外のアミノ酸
            decode_table[i + 3] = amino_acid

        self.encode_table = encode_table
        self.decode_table = decode_table

    def encode(self, texts):
        """ 単語のリストを整数のリストに変換し，クラストークンを付加

        Arg:
            texts(list, ndarray): スペース区切りの単語のリスト
                e.g. ['M N R K K P', 'F L V S Q T', ...]


        Return:
            ndarray: shape=(n_samples, n_words + 1)
                e.g. [[1, 8, 5, 6, 3, 4, 4],
                      [1, 9, 5, 7, 4, 4, 8], ...]

        """
        if type(texts).__module__ == 'numpy':
            texts = np.squeeze(texts)
            texts = texts.tolist()

        texts = [text.split() for text in texts]

        encoded_texts = [self._encode_one(text) for text in texts]
        encoded_texts = self._add_class_token(encoded_texts)

        return encoded_texts

    def _encode_one(self, text):
        encoded_text = []
        for word in text:
            try:
                encoded_text.append(self.encode_table[word])
            except KeyError:
                encoded_text.append(self.encode_table['X'])

        return encoded_text

    def _add_class_token(self, sequences):
        """ class token を先頭に付加する

        class token として1を用いる．

        Arg:
            sequences: arraylike
                shape=(n_sequence, len)

        Return:
            ndarray: shape=(n_sequences, len + 1)
        """
        if isinstance(sequences, list):
            sequences = np.array(sequences)

        # class_token = 1
        cls_arr = np.ones((len(sequences), 1))
        sequences = np.hstack([cls_arr, sequences])

        return sequences.astype(np.int64)

    def decode(self, sequences):
        """ 整数のリストを単語のリストに変換する

        Arg:
            sequences(ndarray): 整数の配列
                shape=(n_samples, n_words)

        Return:
            [str]: 単語のリスト

        """
        # class_tokenを削除
        sequences = np.delete(sequences, 0, axis=-1)

        # ndarrayからlistに変換
        sequences = sequences.tolist()

        texts = []
        for sequence in sequences:
            text = [self.decode_table[n] for n in sequence]
            text = [text[0]] + [word[-1] for word in text[1:]]
            text = ''.join(text)
            texts.append(text)

        return texts

def shuffle(x, y, seed):
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)

def make_example(sequence, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'x': tf.train.Feature(
                int64_list=tf.train.Int64List(value=sequence)),
        'y': tf.train.Feature(
                int64_list=tf.train.Int64List(value=label))
    }))

def write_tfrecord(sequences, labels, filename):
    """ tf.data.Datasetに変換 """
    writer = tf.io.TFRecordWriter(filename)
    for sequence, label in zip(sequences, labels):
        ex = make_example(sequence.tolist(), [int(label)])

        writer.write(ex.SerializeToString())
    writer.close()

def load_dataset(filename, batch_size, length, buffer_size=1000):
    def dict2tuple(feat):
        return feat["x"], feat["y"]

    dataset = tf.data.TFRecordDataset(filenames=filename) \
        .shuffle(buffer_size) \
        .batch(batch_size) \
        .apply(
            tf.data.experimental.parse_example_dataset({
                "x": tf.io.FixedLenFeature([length], dtype=tf.int64),
                "y": tf.io.FixedLenFeature([1], dtype=tf.int64)
            })).map(dict2tuple)

    return dataset

def under_sampling(X, y, sampling_strategy=1.0, random_state=0):
    """ アンダーサンプリングを行う

    Args:
        x(ndarray, dataframe): shape=(n_samples, n_features)
        y(ndarray): shape=(n_samples,)
        random_state(int): シード値

    Return:
        x(ndarray): shape=(n_samples_new, n_features)
        y(ndarray): shape=(n_samples_new,)

    """
    rus = RandomUnderSampler(random_state=random_state,
            sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    return X_resampled, y_resampled