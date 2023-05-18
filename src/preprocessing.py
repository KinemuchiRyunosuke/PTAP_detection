import os
import math
import numpy as np
import json
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.under_sampling import RandomUnderSampler


def preprocess_dataset(motif_data, processed_dir,
                       eval_tfrecord_dir,
                       vocab, n_pos_neg_path,
                       val_rate=0.2, seed=1):
    x_train, y_train, x_test, y_test = [], [], [], []
    for data in motif_data:
        virusname = data['virus'].replace(' ', '_')

        # アミノ酸断片データセットを読み込み
        processed_path = os.path.join(processed_dir, f'{virusname}.pickle')
        with open(processed_path, 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)

        x = vocab.encode(x)
        x = add_class_token(x)

        shuffle(x, y, seed)

        # データセットを学習用と検証用に分割
        boundary = math.floor(len(x) * val_rate)

        # 評価用データセットとしてunder-samplingしていないデータを残しておく
        eval_tfrecord_path = os.path.join(
                eval_tfrecord_dir, f'{virusname}.tfrecord')
        os.makedirs(os.path.dirname(eval_tfrecord_path), exist_ok=True)
        write_tfrecord(x[:boundary], y[:boundary], eval_tfrecord_path)

        x_test.append(x[:boundary])
        y_test.append(y[:boundary])
        x_train.append(x[boundary:])
        y_train.append(y[boundary:])

    x_test, x_train = np.vstack(x_test), np.vstack(x_train)
    y_test, y_train = np.hstack(y_test), np.hstack(y_train)

    # undersamplingの比率の計算(陰性を1/10に減らす)
    n_positive = (y_train == 1).sum()
    n_negative = (len(y_train) - n_positive) // 10
    sampling_strategy = n_positive / n_negative

    # 陽性・陰性のサンプル数をjsonファイルとして記憶
    # 学習で class weight を計算するときに利用する
    dict = {'n_positive': int(n_positive), 'n_negative': int(n_negative)}
    with open(n_pos_neg_path, 'w') as f:
        json.dump(dict, f)

    # undersampling
    x_train, y_train = underSampling(x_train, y_train,
                            sampling_strategy=sampling_strategy)
    x_test, y_test = underSampling(x_test, y_test,
                            sampling_strategy=1.0)

    # シャッフル
    shuffle(x_test, y_test, seed)
    shuffle(x_train, y_train, seed)

    return x_train, x_test, y_train, y_test


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
        sequences = pad_sequences(sequences, padding='post', value=0)

        return sequences

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

        for i, seq in enumerate(sequences):
            try:  # 0が存在しない場合はValueError
                pad_idx = seq.index(0)
            except ValueError:
                continue

            sequences[i] = seq[:pad_idx]

        texts = self.tokenizer.sequences_to_texts(sequences)

        for i, text in enumerate(texts):
            text = text.split()
            text = [text[0]] + list(map(lambda word: word[0], text[1:]))
            texts[i] = ''.join(text)

        return texts

def add_class_token(sequences):
    """ class token を先頭に付加する

    class token として0を用いる．

    Arg:
        sequences: list of int
            shape=(n_sequence, len)

    Return:
        ndarray: shape=(n_sequences, len + 1)
    """
    if isinstance(sequences, list):
        sequences = np.array(sequences)

    sequences += 1

    sequences = np.array(sequences)
    mask = (sequences == 1)
    sequences[mask] = 0

    # class_token = 0
    cls_arr = np.zeros((len(sequences), 1))     # shape=(len(sequences), 1)
    sequences = np.hstack([cls_arr, sequences]).astype('int64')

    return sequences

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

def underSampling(x, y, sampling_strategy=1.0, random_state=0):
    """ アンダーサンプリングを行う

    Args:
        x(ndarray): shape=(n_samples, n_features)
        y(ndarray): shape=(n_samples,)
        random_state(int): シード値

    Return:
        x(ndarray): shape=(n_samples_new, n_features)
        y(ndarray): shape=(n_samples_new,)

    """
    rus = RandomUnderSampler(random_state=random_state,
            sampling_strategy=sampling_strategy)
    x_resampled, y_resampled = rus.fit_resample(x, y)

    return x_resampled, y_resampled