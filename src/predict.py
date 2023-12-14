import os
import json
import re
import pickle
import csv
import math
import argparse
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from Bio import SeqIO

from dataset import Dataset
from preprocessing import under_sampling, shuffle, \
        write_tfrecord, load_dataset, Vocab
from models.transformer import BinaryClassificationTransformer


parser = argparse.ArgumentParser()
parser.add_argument('target_motif', help='解析対象とするSLiM')
parser.add_argument('-t', '--test', type=bool, default=False,
                    help='Trueとした場合，テストモードで実行する')
args = parser.parse_args()

# parameters
length = 26
n_gram = True
val_rate = 0.2
eval_rate = 0.2
num_amino_acid = 23
separate_len = 1
rm_positive_neighbor = 3
motif_neighbor = 2
batch_size = 1024
epochs = 50
threshold = 0.5         # 陽性・陰性の閾値
head_num = 8            # Transformerの並列化に関するパラメータ
dropout_rate = 0.04
hopping_num = 3         # Multi-Head Attentionを施す回数
hidden_dim = 184        # 単語ベクトルの次元数
lr = 2.60e-5            # 学習率
beta = 0.5              # Fベータスコアの引数
seed = 1                # データセットをシャッフルするときのseed値

if args.test:
    batch_size = 100000
    epochs = 1
    hopping_num = 1
    hidden_dim = 8

# paths
motif_data_path = 'references/motif_data.json'
fasta_dir = "data/interim/"
processed_dir = "data/processed/"
tfrecord_dir = "data/tfrecord/"
eval_tfrecord_dir = "data/tfrecord/eval/"
train_tfrecord_path = "data/tfrecord/train_dataset.tfrecord"
test_tfrecord_path = "data/tfrecord/test_dataset.tfrecord"
n_pos_neg_path = "references/n_positive_negative.json"
motif_re_path = "references/motif_regular_expression.json"
checkpoint_path = "models/saved_model.pb"
result_on_training_data_path = "reports/result_on_training_data.csv"
dataset_size_path = "reports/dataset_size.csv"
pred_on_training_path = "reports/positive_pred_on_training_virus.csv"
histogram_path = "reports/histogram.png"
pred_on_nontraining_path = "reports/positive_pred_on_nontraining_virus.csv"

def main():
    gpus = tf.config.list_physical_devices(device_type='GPU')
    if len(gpus) > 0:
        print(f">> GPU detected. {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)

    with open(motif_data_path, 'r') as f:
        motif_data = json.load(f)
        motif_data = motif_data[args.target_motif]

    if not finish_making_dataset(motif_data):
        print("================== MAKING DATASET ==================")

        for content in motif_data:
            virus = content['virus'].replace(' ', '_')
            out_dir = os.path.join(processed_dir, virus)

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            dataset = make_dataset(
                    motif_data=motif_data,
                    virus=virus)

            # タンパク質毎にデータセットを保存
            for protein, (x, y) in dataset.items():
                out_path = os.path.join(out_dir, f'{protein}.pickle')

                with open(out_path, 'wb') as f:
                    if args.test:
                        pickle.dump(x[:10000], f)
                        pickle.dump(y[:10000], f)
                    else:
                        pickle.dump(x, f)
                        pickle.dump(y, f)

    vocab = Vocab(separate_len=separate_len)

    if not (os.path.exists(train_tfrecord_path) \
            and os.path.exists(test_tfrecord_path)):
        print("================== PREPROCESSING ===================")

        # データセットの前処理
        x_train, x_test, y_train, y_test = \
            preprocess_dataset(
                motif_data=motif_data,
                vocab=vocab)

        # tf.data.Datasetとして保存
        write_tfrecord(x_test, y_test, test_tfrecord_path)
        write_tfrecord(x_train, y_train, train_tfrecord_path)

    model = create_model()

    if not os.path.exists(checkpoint_path):
        print("================== TRAINING ========================")
        train(model=model,
              seq_length=length - (separate_len - 1) + 1)

    if not os.path.exists(histogram_path):
        print("================== EVALUATION ======================")
        model.load_weights(os.path.dirname(checkpoint_path))
        evaluate(motif_data=motif_data,
                 model=model,
                 seq_length=length - separate_len + 2,
                 vocab=vocab)

    print("=========== PREDICT ON NON-TRAINING DATA ===========")
    predict_on_nontraining_data(model=model,
                                vocab=vocab)

def make_dataset(motif_data, virus):
    # 対象となるウイルスのJSONデータを取得
    data = None
    for content in motif_data:
        if content['virus'].replace(' ', '_') == virus:
            data = content
            break

    fasta_path = os.path.join(fasta_dir, f'{virus}.fasta')
    with open(fasta_path, 'r') as f:
        records = [record for record in SeqIO.parse(f, 'fasta')]

    dataset = Dataset(
            motifs=data['motifs'],
            protein_subnames=data['protein_subnames'],
            length=length,
            separate_len=separate_len,
            rm_positive_neighbor=rm_positive_neighbor,
            motif_neighbor=motif_neighbor)

    dataset = dataset.make_dataset(records, test_mode=args.test)
    return dataset

def finish_making_dataset(motif_data):
    finish = True

    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        dataset_dir = os.path.join(processed_dir, virus)

        for protein in content['protein_subnames'].keys():
            dataset_path = os.path.join(dataset_dir, f'{protein}.pickle')

            if not os.path.exists(dataset_path):
                finish = False

    return finish

def preprocess_dataset(motif_data, vocab):
    columns = ['virus', 'protein', 'seq']
    X_train = pd.DataFrame(columns=columns)
    X_test = pd.DataFrame(columns=columns)
    y_train = np.array([], dtype=np.int)
    y_test = np.array([], dtype=np.int)

    for content in motif_data:
        virusname = content['virus'].replace(' ', '_')

        # アミノ酸断片データセットを読み込み
        for protein in content['protein_subnames'].keys():
            processed_path = os.path.join(os.path.join(
                    processed_dir, virusname), f'{protein}.pickle')

            with open(processed_path, 'rb') as f:
                x = pickle.load(f)
                y = pickle.load(f)

            shuffle(x, y, seed)

            # 評価用データセットとしてunder-samplingしていないデータを残しておく
            n_eval_ds = math.floor(len(x) * eval_rate)
            x_eval, y_eval = x[:n_eval_ds], y[:n_eval_ds]
            x, y = x[n_eval_ds:], y[n_eval_ds:]

            x_eval = vocab.encode(x_eval)

            eval_tfrecord_path = os.path.join(os.path.join(
                    eval_tfrecord_dir, virusname), f'{protein}.tfrecord')
            os.makedirs(os.path.dirname(eval_tfrecord_path), exist_ok=True)
            write_tfrecord(x_eval, y_eval, eval_tfrecord_path)

            # データセットを学習用と検証用に分割
            n_val_ds = math.floor(len(x) * val_rate)

            df_train = pd.DataFrame({
                'virus': [virusname] * (len(x) - n_val_ds),
                'protein': [protein] * (len(x) - n_val_ds),
                'seq': np.squeeze(x[n_val_ds:])})
            X_train = pd.concat((X_train, df_train))
            y_train = np.hstack((y_train, y[n_val_ds:]))

            df_test = pd.DataFrame({
                'virus': [virusname] * n_val_ds,
                'protein': [protein] * n_val_ds,
                'seq': np.squeeze(x[:n_val_ds])})
            X_test = pd.concat((X_test, df_test))
            y_test = np.hstack((y_test, y[:n_val_ds]))

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
    X_train, y_train = under_sampling(X_train, y_train,
                            sampling_strategy=sampling_strategy)
    X_test, y_test = under_sampling(X_test, y_test,
                            sampling_strategy=1.0)

    # 学習データセットの数を集計
    count_dataset(X_train, y_train, motif_data)

    x_train = X_train['seq']
    x_test = X_test['seq']

    x_train, x_test = vocab.encode(x_train), vocab.encode(x_test)

    # シャッフル
    shuffle(x_test, y_test, seed)
    shuffle(x_train, y_train, seed)

    return x_train, x_test, y_train, y_test

def count_dataset(X, y, motif_data):
    """ データセット数を集計する """
    column_combinations = []
    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        for protein in content['protein_subnames'].keys():
            column_combinations.append((virus, protein))

    df = pd.DataFrame(columns=['virus', 'protein', 'n_positive', 'n_negative', 'total'])

    for virus, protein in column_combinations:
        n_positive = len(X[(y == 1) & (X['virus'] == virus) & (X['protein'] == protein)])
        n_negative = len(X[(y == 0) & (X['virus'] == virus) & (X['protein'] == protein)])
        total = n_positive + n_negative

        row = pd.Series([virus, protein, n_positive, n_negative, total],
                index=df.columns)
        df = df.append(row, ignore_index=True)

    df.to_csv(dataset_size_path, index=False)

def create_model():
    """ モデルを定義する """
    # '<PAD>', '<CLS>', 'X' の3つのトークンを含める
    vocab_size = num_amino_acid ** separate_len + 3

    model = BinaryClassificationTransformer(
                vocab_size=vocab_size,
                hopping_num=hopping_num,
                head_num=head_num,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(
                                learning_rate=lr),
                 loss='binary_crossentropy',
                 metrics=[tf.keras.metrics.Precision(
                            thresholds=threshold,
                            name='precision'),
                          tf.keras.metrics.Recall(
                            thresholds=threshold,
                            name='recall')])

    return model

def train(model, seq_length):
    # クラス重みを設定
    with open(n_pos_neg_path, 'r') as f:
        n_pos_neg = json.load(f)

    total = n_pos_neg['n_positive'] + n_pos_neg['n_negative']
    positive_weight = (1/n_pos_neg['n_positive']) * total / 2.0
    negative_weight = (1/n_pos_neg['n_negative']) * total / 2.0
    class_weight = {0: positive_weight, 1: negative_weight}

    train_ds = load_dataset(train_tfrecord_path,
                            batch_size=batch_size,
                            length=seq_length)
    test_ds = load_dataset(test_tfrecord_path,
                            batch_size=batch_size,
                            length=seq_length)

    # 学習
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
                monitor='val_precision', mode='max', patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_precision', mode='max',
                factor=0.2, patience=3)
    ]

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                         os.path.dirname(checkpoint_path),
                         monitor='val_precision',
                         mode='max', save_best_only=True))

    model.fit(x=train_ds,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=test_ds,
            callbacks=callbacks,
            shuffle=True,
            class_weight=class_weight,
            verbose=1)

def evaluate(motif_data, model, seq_length, vocab):
    if os.path.exists(result_on_training_data_path):
        os.remove(result_on_training_data_path)

    if os.path.exists(pred_on_training_path):
        os.remove(pred_on_training_path)

    # 各ウイルス・タンパク質毎に混同行列を計算
    df_cm = pd.DataFrame(columns=['virus', 'protein', 'tn', 'fp', 'fn', 'tp'])
    df_cm.astype({'tn':'int', 'fp':'int', 'fn':'int', 'tp':'int'})

    # 陽性と判定されたデータの情報を保存
    df_pos_pred = pd.DataFrame(columns=['virus', 'protein', 'y_pred', 'y_true', 'seq'])

    for content in motif_data:
        virusname = content['virus'].replace(' ', '_')

        for protein in content['protein_subnames'].keys():
            eval_ds_path = os.path.join(os.path.join(
                    eval_tfrecord_dir, virusname), f'{protein}.tfrecord')

            eval_ds = load_dataset(eval_ds_path,
                                   batch_size=batch_size,
                                   length=seq_length)

            cm = np.zeros((2, 2), dtype=np.int64)
            for x, y_true in eval_ds:   # 評価用データセットをbatchごとに取り出す
                y_pred = model.predict_on_batch(x)
                y_pred = np.squeeze(y_pred)
                y_true = np.squeeze(y_true)

                df_partial = \
                    pd.DataFrame([[virusname] * len(y_pred[y_pred > threshold]),
                                 [protein] * len(y_pred[y_pred > threshold]),
                                 y_pred[y_pred > threshold],
                                 y_true[y_pred > threshold],
                                 vocab.decode(x[y_pred > threshold])],
                                 index=df_pos_pred.columns)
                df_pos_pred = pd.concat([df_pos_pred, df_partial.T])

                y_pred = (y_pred > threshold).astype(int)
                cm += confusion_matrix(y_true, y_pred, labels=[0, 1])

            # 評価結果をDataFrameに保存
            row = pd.Series([virusname, protein,
                                cm[0,0], cm[0,1], cm[1,0], cm[1,1]],
                            index=df_cm.columns)
            df_cm = pd.concat([df_cm, pd.DataFrame(row).T])

    df_cm.to_csv(result_on_training_data_path, index=False)
    df_pos_pred.to_csv(pred_on_training_path)

    # ヒストグラムの描画
    median = df_pos_pred.y_pred.median()
    x = df_pos_pred['y_pred']
    n, _ , _ = plt.hist(x, bins=50)
    plt.vlines(median, 0, max(n), color='red')
    plt.xlabel('output value')
    plt.savefig(histogram_path)

def predict_on_nontraining_data(model, vocab):
    df_pos_pred = pd.DataFrame(columns=['species', 'description', 'output', 'seq'])

    target_species = glob.glob('data/eval/*.fasta')
    target_species = [os.path.basename(species).replace('.fasta', '') \
                      for species in target_species]

    for species in target_species:
        fasta_path = f'data/eval/{species}.fasta'
        with open(fasta_path, 'r') as f:
            records = [record for record in SeqIO.parse(f, 'fasta') \
                        if (len(record.seq) >= length) \
                            & (not 'X' in record.seq)]

        for record in records:
            seqs = []

            # アミノ酸配列を断片化
            for i in range(len(record.seq) - length + 1):
                seq = str(record.seq[i:(i+length)])
                seq = ' '.join(list(seq))
                seqs.append(seq)

            x = vocab.encode(seqs)

            y_pred = model.predict_on_batch(x)
            y_pred = np.squeeze(y_pred)

            x = x[y_pred > threshold]
            y_pred = y_pred[y_pred > threshold]

            df = pd.DataFrame({
                'species': [species] * len(x),
                'description': [record.description] * len(x),
                'output': y_pred,
                'seq': vocab.decode(x)
                })

            df_pos_pred = pd.concat((df_pos_pred, df))

    df_pos_pred.to_csv(pred_on_nontraining_path, index=False)

def apply_regular_expression(df):
    with open(motif_re_path, 'r') as f:
        motif_re = json.load(f)
        pattern = motif_re[args.target_motif]

    pattern = '.*' + pattern + '.*'
    pattern = re.compile(pattern)

    def pattern_match(seq):
        if pattern.match(seq):
            return True
        else:
            return False

    rows_to_drop = ~df['seq'].map(pattern_match)
    df = df.drop(df[rows_to_drop].index, inplace=True)

    return df

if __name__ == '__main__':
    main()
