import os
import json
import shutil
import invoke

motif_data_path = 'references/motif_data.json'
processed_dir = "data/processed/"
eval_tfrecord_dir = "data/tfrecord/eval/"
train_tfrecord_path = "data/tfrecord/train_dataset.tfrecord"
test_tfrecord_path = "data/tfrecord/test_dataset.tfrecord"
vocab_path = "references/vocab.pickle"
n_pos_neg_path = "references/n_positive_negative.json"
model_dir = "models/"
evaluation_path = "reports/result/evaluation.csv"
false_positive_path = "reports/result/false_positive.csv"
positive_pred_path = "reports/result/positive_pred.csv"


@invoke.task
def clear(c):
    """生成されたデータを削除"""
    with open(motif_data_path, 'r') as f:
        motif_data = json.load(f)

    # data/processed/ のファイルを削除
    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        out_path = os.path.join(processed_dir, f'{virus}.pickle')
        remove(out_path)

    # data/tfrecord/ のファイルを削除
    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        out_path = os.path.join(eval_tfrecord_dir, f'{virus}.tfrecord')
        remove(out_path)

    remove(train_tfrecord_path)
    remove(test_tfrecord_path)

    # reference/ のファイルを削除
    remove(vocab_path)
    remove(n_pos_neg_path)

    # models/ のファイルを削除
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        invoke.run('touch {}/.gitkeep'.format(model_dir))

    # reports/ のファイルを削除
    remove(evaluation_path)
    remove(false_positive_path)
    remove(positive_pred_path)

def remove(path):
    if os.path.exists(path):
        os.remove(path)
