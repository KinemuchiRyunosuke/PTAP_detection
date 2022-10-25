import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from preprocessing import load_dataset


def evaluate_precision(motif_data, model, seq_length, batch_size, threshold,
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
