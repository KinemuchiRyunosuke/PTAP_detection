import argparse
import os
import numpy as np
import tensorflow as tf
import pickle
import json
import pandas as pd
import csv

from models.transformer import BinaryClassificationTransformer
from sklearn.metrics import confusion_matrix

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
parser.add_argument('false_positive_path', type=str)

args = parser.parse_args()


def main():
    json_path = 'references/PTAP_data.json'
    with open(json_path, 'r') as f:
       json_data = json.load(f)

    model = create_model()
    model.load_weights(args.checkpoint_path)

    df = pd.DataFrame(columns=['virus', 'protein', 'tn', 'fp', 'fn', 'tp'])
    for data in json_data:
        virusname = data['virus'].replace(' ', '_')

        for protein in data['proteins']:
            cm = np.zeros((2, 2))
            eval_ds_path = os.path.join(os.path.join(
                    args.eval_tfrecord_dir, virusname),
                    f'{protein}.tfrecord')

            # 最適な閾値で評価
            eval_ds = load_dataset(eval_ds_path,
                                   batch_size=args.batch_size,
                                   length=args.length+1)
            
            for x, y_true in eval_ds:
                y_true = np.squeeze(y_true)
                y_pred = model.predict(x)
                y_pred = np.squeeze(y_pred)
                y_pred = (y_pred >= args.threshold).astype(int)
                cm_batch = confusion_matrix(y_true, y_pred, labels=[0, 1])
                cm += cm_batch

                with open(args.vocab_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                vocab = Vocab(tokenizer)

                # 偽陽性データを保存
                x_fp = x[(y_true == 0) & (y_pred == 1)]
                x_fp = vocab.decode(x_fp, class_token=True)

                for seq in x_fp:
                    with open(args.false_positive_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([virusname, protein, seq])

            # 評価結をDataFrameに保存
            row = pd.Series([virusname, protein,
                    cm[0,0], cm[0,1], cm[1,0], cm[1,1]], index=df.columns)
            df = df.append(row, ignore_index=True)

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


if __name__ == '__main__':
    main()
