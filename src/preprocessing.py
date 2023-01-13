import os
import math
import numpy as np
import json
import pickle
import itertools

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.under_sampling import RandomUnderSampler


def preprocess_dataset(motif_data, processed_dir,
                       train_tfrecord_path, test_tfrecord_path,
                       eval_tfrecord_dir,
                       vocab, n_pos_neg_path,
                       val_rate=0.2, seed=1):
    x_train, y_train, x_test, y_test = [], [], [], []
    for data in motif_data:
        virusname = data['virus'].replace(' ', '_')
        dataset_dir = os.path.join(processed_dir, virusname)

        for protein in data['proteins'].keys():
            # アミノ酸断片データセットを読み込み
            processed_path = os.path.join(dataset_dir, f'{protein}.pickle')
            with open(processed_path, 'rb') as f:
                x = pickle.load(f)
                y = pickle.load(f)

            x = vocab.encode(x)

            shuffle(x, y, seed)

            # データセットを学習用と検証用に分割
            boundary = math.floor(len(x) * val_rate)

            def attention_mask(n_samples, n_dim):
                return np.ones((n_samples, n_dim), dtype=int).tolist()

            # 評価用データセットとしてunder-samplingしていないデータを残しておく
            eval_tfrecord_path = os.path.join(os.path.join(
                    eval_tfrecord_dir, virusname), f'{protein}.tfrecord')
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

    # tf.data.Datasetとして保存
    write_tfrecord(x_test, y_test, test_tfrecord_path)
    write_tfrecord(x_train, y_train, train_tfrecord_path)

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
        ex = make_example(sequence, [int(label)])

        writer.write(ex.SerializeToString())
    writer.close()


class Vocab:
    """ アミノ酸配列をIDに変換したり，IDをアミノ酸配列に直したりする """
    def __init__(self, separate_len, class_token=False,
                 include_X=False):
        self._amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        if include_X:
            self._amino_acids.append('X')

        # list of str: <PAD>や<CLS>のようなトークンを保持
        self.tokens = ['<UNK>']

        # int: 単語ベクトルひとつあたりのアミノ酸の数
        self.separate_len = separate_len

        # bool: Trueのとき，先頭にclassトークンを追加する
        self.class_token = class_token

        if self.class_token:
            self.tokens.append('<CLS>')

        self.amino_acids_dict, self.amino_acids_dict_reverse = \
                self._mk_amino_acids_dict()

    @property
    def num_tokens(self):
        return len(self.tokens)

    def encode(self, seqs):
        """
        アミノ酸配列をIDに変換

        Arg:
            seqs(list of str or ndarray): アミノ酸配列

        Return:
            list of int: ID

        """
        seqs = np.squeeze(seqs).reshape(-1)

        def to_id(word):
            try:
                id = self.amino_acids_dict[word]
            except KeyError:
                id = self.tokens.index('<UNK>')
            return id

        encoded_seqs = []
        for seq in seqs:
            words = []
            for i in range(len(seq) - self.separate_len + 1):
                words.append(seq[i:(i + self.separate_len)])

            encoded_seqs.append(list(map(to_id, words)))

        if self.class_token:
            encoded_seqs = self._add_class_token(encoded_seqs)

        return encoded_seqs

    def decode(self, seqs):
        """
        IDをアミノ酸配列に変換

        Arg:
            seqs: ID配列
        """
        def to_amino_acids(id):
            try:
                word = self.amino_acids_dict_reverse[id]
            except KeyError:
                word = 'X'
            return word

        decoded_seqs = []
        for seq in seqs:
            if self.class_token:
                seq = seq[1:]

            words = map(to_amino_acids, seq)
            head = words.__next__()
            words = map(lambda word: word[-1], words)

            decoded_seq = head + ''.join(words)
            decoded_seqs.append(decoded_seq)

        return decoded_seqs

    def _mk_amino_acids_dict(self):
        amino_acids_dict, amino_acids_dict_reverse = {}, {}

        amino_acids_permutations = [''.join(permutations) for permutations
                in itertools.product(self._amino_acids, repeat=self.separate_len)]

        for i, amino_acid in enumerate(amino_acids_permutations):
            amino_acids_dict[amino_acid] = i + self.num_tokens
            amino_acids_dict_reverse[i + self.num_tokens] = amino_acid

        return amino_acids_dict, amino_acids_dict_reverse

    def _add_class_token(self, seqs):
        class_id = self.tokens.index('<CLS>')

        def add_class_token_one(seq):
            return [class_id] + seq

        return list(map(add_class_token_one, seqs))

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