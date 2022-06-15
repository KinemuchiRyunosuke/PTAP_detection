import argparse
import os
import numpy as np
import tensorflow as tf
import pickle
import json
import pandas as pd

from models.transformer import BinaryClassificationTransformer

from features.preprocessing import Vocab, load_dataset


# コマンドライン引数を取得
parser = argparse.ArgumentParser()

parser.add_argument('length', type=int)
parser.add_argument('batch_size', type=int)
parser.add_argument('num_words', type=int)
parser.add_argument('hopping_num', type=int)
parser.add_argument('head_num', type=int)
parser.add_argument('hidden_dim', type=int)
parser.add_argument('dropout_rate', type=float)
parser.add_argument('lr', type=float)
parser.add_argument('beta', type=float)
parser.add_argument('threshold', type=float)

parser.add_argument('checkpoint_path', type=str)
parser.add_argument('eval_tfrecord_dir', type=str)
parser.add_argument('vocab_path', type=str)
parser.add_argument('result_path', type=str)
parser.add_argument('false_positive_dir', type=str)

args = parser.parse_args()


def main():
    json_path = 'references/PTAP_data.json'
    with open(json_path, 'r') as f:
       json_data = json.load(f)

    model = create_model()
    model.load_weights(args.checkpoint_path)

    df = pd.DataFrame(columns=['virus', 'protein', 'tn', 'fn', 'fp', 'tp'])
    for data in json_data:
        virusname = data['virus'].replace(' ', '_')

        for protein in data['proteins']:
            eval_ds_path = os.path.join(os.path.join(
                    args.eval_tfrecord_dir, virusname),
                    f'{protein}.tfrecord')
            fp_path = os.path.join(os.path.join(
                args.false_positive_dir, virusname), f'{protein}.txt')

            # 最適な閾値で評価
            eval_ds = load_dataset(eval_ds_path,
                                   batch_size=args.batch_size,
                                   length=args.length+1)

            ys_pred = model.predict(eval_ds)
            ys_pred = np.squeeze(ys_pred)
            cm = calc_confusion_matrix(ys_pred, eval_ds, args.threshold)

            # 評価結果をDataFrameに保存
            row = pd.Series([virusname, protein,
                    cm[0,0], cm[0,1], cm[1,0], cm[1,1]], index=df.columns)
            df = df.append(row, ignore_index=True)

            with open(args.vocab_path, 'rb') as f:
                tokenizer = pickle.load(f)
            vocab = Vocab(tokenizer)

            # 偽陽性データを保存
            for i, (x, y_true) in enumerate(eval_ds):
                y_true = y_true.numpy()
                y_true = np.squeeze(y_true)
                y_pred = ys_pred[i:(i + y_true.shape[0])]
                y_pred = (y_pred >= args.threshold).astype(int)

                x_fp = x[(y_true == 0) & (y_pred == 1)]
                x_fp = vocab.decode(x_fp, class_token=True)

                for seq in x_fp:
                    with open(fp_path, 'a') as f:
                        print(seq, file=f)

    df.to_csv(args.result_path)


def create_model():
    """ モデルを定義する """
    model = BinaryClassificationTransformer(
                vocab_size=args.num_words,
                hopping_num=args.hopping_num,
                head_num=args.head_num,
                hidden_dim=args.hidden_dim,
                dropout_rate=args.dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(
                                learning_rate=args.lr),
                 loss='binary_crossentropy')

    return model

def calc_confusion_matrix(ys_pred, ds, threshold):
    cm = np.zeros((2, 2)).astype(int)
    for i, (_, y_true) in enumerate(ds):
        y_true = y_true.numpy()
        y_true = np.squeeze(y_true)
        y_pred = ys_pred[i:(i + y_true.shape[0])]
        y_pred = (y_pred >= threshold).astype(int)

        cm[0, 0] += ((y_true == 0) & (y_pred == 0)).astype(int).sum()
        cm[0, 1] += ((y_true == 0) & (y_pred == 1)).astype(int).sum()
        cm[1, 0] += ((y_true == 1) & (y_pred == 0)).astype(int).sum()
        cm[1, 1] += ((y_true == 1) & (y_pred == 1)).astype(int).sum()

    return cm

def calc_precision(cm):
    if cm[1, 1] != 0:
        precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    else:
        precision = 0
    return precision

def calc_recall(cm):
    if cm[1, 1] != 0:
        recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    else:
        recall = 0
    return recall


if __name__ == '__main__':
    main()
