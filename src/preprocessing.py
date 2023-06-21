import numpy as np

import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler


class Vocab:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, texts):
        """ 単語のベクトル化の準備

        Arg:
            texts([str], ndarray): 単語のリスト
                e.g. ['VPTAPP',
                      'ATSQVP']

        Return:
            self(Vocab instance): 学習を完了したインスタンス

        """
        if type(texts).__module__ == 'numpy':
            texts = np.squeeze(texts)
            texts = texts.tolist()

        self.tokenizer.fit_on_texts(texts)
        return self

    def encode(self, texts):
        """ 単語のリストを整数のリストに変換する

        Arg:
            texts(list, ndarray): 単語のリスト

        Return:
            ndarray: shape=(n_samples, n_words)
                e.g. [[0, 1, 2, 3, 4, 4],
                      [3, 2, 5, 6, 0, 4]]

        """
        if type(texts).__module__ == 'numpy':
            texts = np.squeeze(texts)
            texts = texts.tolist()

        sequences = self.tokenizer.texts_to_sequences(texts)

        return np.array(sequences, dtype=np.int64)

    def decode(self, sequences, class_token=False):
        """ 整数のリストを単語のリストに変換する

        Arg:
            sequences(ndarray): 整数の配列
                shape=(n_samples, n_words)

        Return:
            [str]: 単語のリスト

        """
        if class_token:  # class_tokenを削除
            sequences = np.delete(sequences, 0, axis=-1)

        # ndarrayからlistに変換
        sequences = sequences.tolist()

        texts = self.tokenizer.sequences_to_texts(sequences)

        for i, text in enumerate(texts):
            text = text.split()
            text = [text[0]] + list(map(lambda word: word[0], text[1:]))
            texts[i] = ''.join(text)

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

def add_class_token(sequences):
    """ class token を先頭に付加する

    class token として0を用いる．

    Arg:
        sequences: arraylike
            shape=(n_sequence, len)

    Return:
        ndarray: shape=(n_sequences, len + 1)
    """
    if isinstance(sequences, list):
        sequences = np.array(sequences)

    # class_token = 0
    cls_arr = np.zeros((len(sequences), 1))
    sequences = np.hstack([cls_arr, sequences])

    return sequences.astype(np.int64)

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