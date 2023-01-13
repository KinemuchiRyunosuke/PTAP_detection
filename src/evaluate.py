import os
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix

from preprocessing import Vocab, load_dataset


def evaluate(motif_data, model, seq_length, batch_size, threshold,
             eval_tfrecord_dir, vocab, result_path,
             false_positive_path, positive_pred_path):
    if os.path.exists(false_positive_path):
        os.remove(false_positive_path)

    if os.path.exists(positive_pred_path):
        os.remove(positive_pred_path)

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

            cm = np.zeros((2, 2))
            for x, y_true in eval_ds:
                y_pred = model.predict_on_batch(x)
                y_pred = y_pred['logits']
                y_pred = y_pred.argmax(axis=1)
                y_true = np.squeeze(y_true)
                cm += confusion_matrix(y_true, y_pred, labels=[0, 1])

                # 偽陽性データを保存
                x_fp = x[(y_true == 0) & (y_pred == 1)]
                x_fp = vocab.decode(x_fp.numpy())

                x_pos_pred = x[(y_pred == 1)]
                x_pos_pred = vocab.decode(x_pos_pred.numpy())

                for seq in x_fp:
                    with open(false_positive_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([virusname, protein, seq])

                for seq in x_pos_pred:
                    with open(positive_pred_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([virusname, protein, seq])

            # 評価結果をDataFrameに保存
            row = pd.Series([virusname, protein,
                    cm[0,0], cm[0,1], cm[1,0], cm[1,1]], index=df.columns)
            df = pd.concat([df, pd.DataFrame(row).T])

    df.to_csv(result_path)
