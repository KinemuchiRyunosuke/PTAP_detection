import os
import json
import shutil
import invoke

motif_data_path = 'references/motif_data.json'
processed_dir = "data/processed/"
eval_tfrecord_dir = "data/tfrecord/eval/"
train_tfrecord_path = "data/tfrecord/train_dataset.tfrecord"
test_tfrecord_path = "data/tfrecord/test_dataset.tfrecord"
tfrecord_dir = 'data/tfrecord/'
vocab_path = "references/vocab.pickle"
n_pos_neg_path = "references/n_positive_negative.json"
model_dir = "models/"
evaluation_path = "reports/result/evaluation.csv"
false_positive_path = "reports/result/false_positive.csv"
positive_pred_path = "reports/result/positive_pred.csv"
best_params = 'reports/result/best_parameters.txt'


@invoke.task
def clear(c):
    """生成されたデータを削除"""
    # data/processed/ のファイルを削除
    remove_dir(processed_dir)

    # data/tfrecord/ のファイルを削除
    remove_dir(tfrecord_dir)

    # reference/ のファイルを削除
    remove(vocab_path)
    remove(n_pos_neg_path)

    # models/ のファイルを削除
    remove_dir(model_dir)

    # reports/ のファイルを削除
    remove(evaluation_path)
    remove(false_positive_path)
    remove(positive_pred_path)
    remove(best_params)

    # data/separate_len/ のファイルを削除
    for separate_len in range(5):
        remove_dir('data/separate_len{}'.format(separate_len))

def remove(path):
    if os.path.exists(path):
        os.remove(path)

def remove_dir(dir_path):
    has_gitkeep = os.path.exists('{}/.gitkeep'.format(dir_path))

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

        if has_gitkeep:
            invoke.run('touch {}/.gitkeep'.format(dir_path))