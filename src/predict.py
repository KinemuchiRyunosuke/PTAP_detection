import os
import copy
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
import datetime

from sklearn.metrics import confusion_matrix, precision_recall_curve
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
test_rate = 0.2     # 評価用データセットの割合
val_rate = 0.2      # 非評価用データセットにおける検証用データセットの割合
num_amino_acid = 23
separate_len = 1
rm_positive_neighbor = 0
motif_neighbor = 0
batch_size = 1024
epochs = 20
threshold = 0.5         # 陽性・陰性の閾値
head_num = 8            # Transformerの並列化に関するパラメータ
dropout_rate = 0.04
hopping_num = 3         # Multi-Head Attentionを施す回数
hidden_dim = 184        # 単語ベクトルの次元数
lr = 2.60e-5            # 学習率
beta = 0.1              # Fベータスコアの引数
seed = 1                # データセットをシャッフルするときのseed値
eval_threshold = 0

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
test_tfrecord_dir = "data/tfrecord/test/"
train_tfrecord_path = "data/tfrecord/train_dataset.tfrecord"
val_tfrecord_path = "data/tfrecord/val_dataset.tfrecord"
motif_re_path = "references/motif_regular_expression.json"
checkpoint_path = "models/saved_model.pb"
performance_evaluation_path = "reports/performance_evaluation.csv"
dataset_size_path = "reports/dataset_size.csv"
ys_pred_true_path = "reports/ys_pred_true.pickle"
pred_on_val_path = "reports/pred_on_val.csv"
pred_on_nontraining_path = "reports/positive_pred_on_nontraining_virus.csv"
parameters_path = "reports/parameters.json"


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
            and os.path.exists(val_tfrecord_path)):
        print("================== PREPROCESSING ===================")
        preprocess_dataset(motif_data=motif_data, vocab=vocab)

    model = create_model()

    if not os.path.exists(checkpoint_path):
        print("================== TRAINING ========================")
        train(model=model,
              seq_length=length - (separate_len - 1) + 1)

    if not os.path.exists(pred_on_val_path):
        print("================== EVALUATION ======================")
        model.load_weights(os.path.dirname(checkpoint_path))
        evaluate(motif_data=motif_data,
                 model=model,
                 seq_length=length - separate_len + 2,
                 vocab=vocab)

    # パラメータを保存
    save_parameters()

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
    # アミノ酸配列・ラベルをDataFrameとして取り出す
    df = pd.DataFrame(columns=['virus', 'protein', 'seq', 'label'])
    for content in motif_data:
        virusname = content['virus'].replace(' ', '_')

        for protein in content['protein_subnames'].keys():
            processed_path = os.path.join(os.path.join(
                    processed_dir, virusname), f'{protein}.pickle')

            with open(processed_path, 'rb') as f:
                x = pickle.load(f)
                y = pickle.load(f)

            df_partial = pd.DataFrame({
                'virus': [virusname] * len(x),
                'protein': [protein] * len(x),
                'seq': x,
                'label': y})
            df = pd.concat((df, df_partial))

    # 評価用データセットを各タンパク質ごとに等しい割合で抽出
    df_test, df = divide_df(df, keys=['virus', 'protein'], rate=test_rate)

    # 評価用データセットをウイルス・タンパク質毎に保存
    for (virus, protein), group in df_test.groupby(['virus', 'protein']):
        test_tfrecord_path = os.path.join(os.path.join(
                test_tfrecord_dir, virus), f'{protein}.tfrecord')
        os.makedirs(os.path.dirname(test_tfrecord_path), exist_ok=True)
        write_tfrecord(vocab.encode(group['seq']),
                       group['label'].to_numpy(dtype=int),
                       test_tfrecord_path)

    # 検証用データと訓練用データを分割
    df_val, df_train = divide_df(df, keys=['virus', 'protein'], rate=val_rate)

    # 学習データセットの数を集計
    count_dataset(df_train)

    df_train = get_sample_weights(df_train)

    y_train = df_train['label'].to_numpy(dtype=int)
    y_val = df_val['label'].to_numpy(dtype=int)

    # undersamplingの比率の計算(陰性を1/10に減らす)
    n_positive = (y_train == 1).sum()
    n_negative = (len(y_train) - n_positive) // 10
    sampling_strategy = n_positive / n_negative

    # undersampling
    df_train, y_train = under_sampling(df_train, y_train,
                            sampling_strategy=sampling_strategy)
    df_val, y_val = under_sampling(df_val, y_val,
                            sampling_strategy=1.0)

    x_train = vocab.encode(df_train['seq'])
    x_val = vocab.encode(df_val['seq'])
    sample_weights = df_train['sample_weight'].to_numpy(dtype=float)

    # tfrecordを保存
    write_tfrecord(x_val, y_val, val_tfrecord_path)
    write_tfrecord(x_train, y_train, train_tfrecord_path,
                   sample_weights=sample_weights)

def divide_df(df, keys, rate):
    """ DataFrameをグループごとに等しい割合で2つに分割する

    Args:
        df(pd.DataFrame): 分割前のDataFrame
        keys(str or list): グループ分けを行う列名
        rate(floot): DataFrameをdf_Aとdf_Bのふたつに分割する
            としたとき，df_AのDataFrameの割合を指定する．

    Returns:
        (df_A(pd.DataFrame), df_B(pd.DataFrame)):
            分割後のDataFrame.

    """
    df_A = pd.DataFrame(columns=df.columns)
    df_B = pd.DataFrame(columns=df.columns)

    for _, group in df.groupby(keys):
        group = group.sample(frac=1)    # シャッフル
        n = math.floor(len(group) * rate)
        df_A = pd.concat((df_A, group[:n]))
        df_B = pd.concat((df_B, group[n:]))

    return df_A, df_B

def count_dataset(df):
    """ データセット数を集計する """
    def n_positive(d):
        return (d.label == 1).sum()

    table = pd.DataFrame({
        'total': df.groupby(['virus', 'protein']).size(),
        'n_positive': df.groupby(['virus', 'protein']).apply(n_positive)
    })

    table['n_negative'] = table['total'] - table['n_positive']

    table.to_csv(dataset_size_path, index=True)

def get_sample_weights(df):
    """ 入力されたDataFrameにsample weightの列を追加 """
    total = len(df)
    n_positive = (df['label'] == 1).sum()
    n_negative = total - n_positive
    n_virus = len(df['virus'].unique())

    df['sample_weight'] = df.groupby('virus')['label'].transform(
            lambda s: n_positive / (n_virus * (s == 1).sum()))
    df.loc[df['label'] == 0, 'sample_weight'] = 1

    positive_weight = total / (2.0 * n_positive)
    negative_weight = total / (2.0 * n_negative)
    df.loc[df['label'] == 1, 'sample_weight'] *= positive_weight
    df.loc[df['label'] == 0, 'sample_weight'] *= negative_weight

    return df

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
    train_ds = load_dataset(train_tfrecord_path,
                            batch_size=batch_size,
                            length=seq_length)
    val_ds = load_dataset(val_tfrecord_path,
                            batch_size=batch_size,
                            length=seq_length)

    # 学習
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
                monitor='val_precision', mode='max', patience=2),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_precision', mode='max',
                factor=0.2, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
                os.path.dirname(checkpoint_path),
                monitor='val_precision',
                mode='max',
                save_best_only=True)
    ]

    model.fit(x=train_ds,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            shuffle=True,
            verbose=1)

def evaluate(motif_data, model, seq_length, vocab):
    if os.path.exists(performance_evaluation_path):
        os.remove(performance_evaluation_path)

    if os.path.exists(pred_on_val_path):
        os.remove(pred_on_val_path)

    # 陽性と判定されたデータの情報を保存
    df = pd.DataFrame(columns=['virus', 'protein', 'y_pred', 'y_true', 'seq'])
    for content in motif_data:
        virusname = content['virus'].replace(' ', '_')

        for protein in content['protein_subnames'].keys():
            test_ds_path = os.path.join(os.path.join(
                    test_tfrecord_dir, virusname), f'{protein}.tfrecord')

            test_ds = load_dataset(test_ds_path,
                                   batch_size=batch_size,
                                   length=seq_length)

            cm = np.zeros((2, 2), dtype=np.int64)
            for x, y_true in test_ds:   # 評価用データセットをbatchごとに取り出す
                y_pred = model.predict_on_batch(x)
                y_pred = np.squeeze(y_pred)
                y_true = np.squeeze(y_true)

                df_partial = \
                    pd.DataFrame([[virusname] * len(y_pred),
                                 [protein] * len(y_pred),
                                 y_pred,
                                 y_true,
                                 vocab.decode(x)],
                                 index=df.columns)
                df = pd.concat([df, df_partial.T])

    df.to_csv(pred_on_val_path)

    # f1スコアを最適化する閾値を計算
    precision, recall, threshold_from_pr = precision_recall_curve(
            y_true=df['y_true'].to_numpy().astype(np.float32),
            probas_pred=df['y_pred'].to_numpy().astype(np.float32))
    a = (1 + beta ** 2) * precision * recall
    b = (beta ** 2) * precision + recall
    f_beta = np.divide(a, b, out=np.zeros_like(a), where=b!=0)

    global eval_threshold
    eval_threshold = np.max(f_beta)

    # 各ウイルス・タンパク質毎に混同行列を計算
    df_cm = pd.DataFrame(columns=['virus', 'protein', 'tn', 'fp', 'fn', 'tp'])
    df_cm.astype({'tn':'int', 'fp':'int', 'fn':'int', 'tp':'int'})

    for (virus, protein), group in df.groupby(['virus', 'protein']):
        cm = confusion_matrix(group['y_true'].to_numpy().astype(int),
                              (group['y_pred'] > eval_threshold).astype(int),
                              labels=[0, 1])
        row = pd.Series([virus, protein, cm[0,0], cm[0,1], cm[1,0], cm[1,1]],
                        index=df_cm.columns)
        df_cm = pd.concat([df_cm, pd.DataFrame(row).T])

    df_cm.to_csv(performance_evaluation_path)

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

            x = x[y_pred > eval_threshold]
            y_pred = y_pred[y_pred > eval_threshold]

            df = pd.DataFrame({
                'species': [species] * len(x),
                'description': [record.description] * len(x),
                'output': y_pred,
                'seq': vocab.decode(x)
                })

            df_pos_pred = pd.concat((df_pos_pred, df))

    df_pos_pred.to_csv(pred_on_nontraining_path, index=False)

def save_parameters():
    parameters = {}

    # 現在時刻を保存
    parameters['date'] = str(datetime.datetime.now())

    parameters['length'] = globals()['length']
    parameters['n_gram'] = globals()['n_gram']
    parameters['test_rate'] = globals()['test_rate']
    parameters['val_rate'] = globals()['val_rate']
    parameters['num_amino_acid'] = globals()['num_amino_acid']
    parameters['separate_len'] = globals()['separate_len']
    parameters['rm_positive_neighbor'] = globals()['rm_positive_neighbor']
    parameters['motif_neighbor'] = globals()['motif_neighbor']
    parameters['batch_size'] = globals()['batch_size']
    parameters['epochs'] = globals()['epochs']
    parameters['threshold'] = globals()['threshold']
    parameters['head_num'] = globals()['head_num']
    parameters['dropout_rate'] = globals()['dropout_rate']
    parameters['hopping_num'] = globals()['hopping_num']
    parameters['hidden_dim'] = globals()['hidden_dim']
    parameters['lr'] = globals()['lr']
    parameters['beta'] = globals()['beta']
    parameters['seed'] = globals()['seed']
    parameters['eval_threshold'] = globals()['eval_threshold']

    with open(parameters_path, 'w') as f:
        json.dump(parameters, f, indent=2)


if __name__ == '__main__':
    main()
