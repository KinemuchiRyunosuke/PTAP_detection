import os
import numpy as np
import json
import pickle
import optuna
import tensorflow as tf
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
from preprocessing import load_dataset

from dataset import make_dataset
from fit_vocab import fit_vocab
from preprocessing import preprocess_dataset, write_tfrecord
from train_model import train
from models.transformer import BinaryClassificationTransformer


# parameters
n_gram = True
val_rate = 0.2
num_amino_acid = 22
batch_size = 1024
epochs = 50
threshold = 0.5         # 陽性・陰性の閾値
head_num = 8            # Transformerの並列化に関するパラメータ
dropout_rate = 0.04
beta = 0.5              # Fベータスコアの引数
seed = 1                # データセットをシャッフルするときのseed値
n_trials = 100

# TEST===========================================================
# batch_size = 100000
# epochs = 1
# hopping_num = 1
# hidden_dim = 8
# n_trials = 3
# ===============================================================

# paths
motif_data_path = 'references/motif_data.json'
fasta_dir = "data/interim/"
processed_dir = "data/separate_len{}/processed/"
tfrecord_dir = "data/separate_len{}/tfrecord/"
eval_tfrecord_dir = "data/separate_len{}/tfrecord/eval/"
train_tfrecord_path = "data/separate_len{}/tfrecord/train_dataset.tfrecord"
test_tfrecord_path = "data/separate_len{}/tfrecord/test_dataset.tfrecord"
vocab_path = "references/vocab.pickle"
n_pos_neg_path = "references/n_positive_negative.json"
model_dir = "models/"
checkpoint_path = "models/saved_model.pb"
best_parameters_path = "reports/result/best_parameters.txt"


def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    with open(best_parameters_path, 'w') as f:
        print("best_parameters:", file=f)
        print(study.best_params, file=f)
        print("best_score:", file=f)
        print(study.best_value, file=f)


def objective(trial):
    with open(motif_data_path, 'r') as f:
        motif_data = json.load(f)

    length = trial.suggest_int('length', 20, 34)
    lr = trial.suggest_float('learning_rate', 1e-8, 1, log=True)
    hopping_num = trial.suggest_int('hopping_num', 1, 6)
    hidden_dim = 8 * trial.suggest_int('hidden_dim/8', 4, 124, log=True)
    separate_len = trial.suggest_int('separate_len', 1,
            min(4, math.ceil(4 * 250 / hidden_dim)))    # メモリ不足予防のために上限設定
    num_words = num_amino_acid ** separate_len + 3
    seq_length = length - separate_len + 2

    # データセット作成
    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        out_dir = os.path.join(processed_dir.format(separate_len), virus)
        dataset = make_dataset(
                motif_data=motif_data,
                length=length,
                virus=virus,
                fasta_dir=fasta_dir,
                separate_len=separate_len)

        # TEST======================================================
        # for key, (x, y) in dataset.items():
        #     dataset[key] = (x[:1000], y[:1000])
        #===========================================================

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # データセットを保存
        for protein, (x, y) in dataset.items():
            out_path = os.path.join(out_dir, f'{protein}.pickle')
            with open(out_path, 'wb') as f:
                pickle.dump(x, f)
                pickle.dump(y, f)

    vocab = fit_vocab(motif_data=motif_data,
                      num_words=num_words,
                      dataset_dir=processed_dir.format(separate_len),
                      vocab_path=vocab_path)

    # データセットの前処理
    x_train, x_test, y_train, y_test = \
        preprocess_dataset(
            motif_data=motif_data,
            processed_dir=processed_dir.format(separate_len),
            eval_tfrecord_dir=eval_tfrecord_dir.format(separate_len),
            vocab=vocab,
            n_pos_neg_path=n_pos_neg_path,
            val_rate=val_rate,
            seed=seed)

    # tf.data.Datasetとして保存
    write_tfrecord(x_test, y_test,
            test_tfrecord_path.format(separate_len))
    write_tfrecord(x_train, y_train,
            train_tfrecord_path.format(separate_len))

    model = create_model(num_words, hopping_num, hidden_dim, lr)

    train(model=model,
          seq_length=seq_length,
          batch_size=batch_size,
          epochs=epochs,
          n_pos_neg_path=n_pos_neg_path,
          train_tfrecord_path=train_tfrecord_path.format(separate_len),
          test_tfrecord_path=test_tfrecord_path.format(separate_len))

    f_beta_score = evaluate(
            motif_data=motif_data,
            model=model,
            seq_length=seq_length,
            batch_size=batch_size,
            threshold=threshold,
            eval_tfrecord_dir=eval_tfrecord_dir.format(separate_len),
            beta=beta)

    return -f_beta_score

def create_model(num_words, hopping_num, hidden_dim, lr):
    """ モデルを定義する """
    model = BinaryClassificationTransformer(
                vocab_size=num_words,
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

def evaluate(motif_data, model, seq_length, batch_size, threshold,
             eval_tfrecord_dir, beta):
    df = pd.DataFrame(columns=['virus', 'protein', 'tn', 'fp', 'fn', 'tp'])
    for data in motif_data:
        virusname = data['virus'].replace(' ', '_')

        for protein in data['proteins']:
            eval_ds_path = os.path.join(os.path.join(
                    eval_tfrecord_dir, virusname),
                    f'{protein}.tfrecord')

            # 最適な閾値で評価
            eval_ds = load_dataset(eval_ds_path,
                                   batch_size=batch_size,
                                   length=seq_length)

            for x, y_true in eval_ds:
                y_pred = model.predict(x)
                y_pred = np.squeeze(y_pred)
                y_pred = (y_pred > threshold).astype(int)
                y_true = np.squeeze(y_true)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                # 評価結果をDataFrameに保存
                row = pd.Series([virusname, protein,
                        cm[0,0], cm[0,1], cm[1,0], cm[1,1]], index=df.columns)
                df = pd.concat([df, pd.DataFrame(row).T])

    # precisionの計算
    total_fp = df['fp'].sum()
    total_fn = df['fn'].sum()
    total_tp = df['tp'].sum()

    return f_beta_score(total_tp, total_fp, total_fn, beta)

def f_beta_score(tp, fp, fn, beta):
    try:
        f_beta_score = (1 + beta**2) * tp / \
                ((1 + beta**2) * tp + beta**2 * fn + fp)
    except ZeroDivisionError:
        f_beta_score = 0

    return f_beta_score


if __name__ == '__main__':
    main()