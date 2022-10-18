import os
import json
import shutil
import invoke

motif_data_path = 'references/PTAP_data.json'
processed_dir = "data/processed/"
eval_tfrecord_dir = "data/tfrecord/eval/"
train_tfrecord_path = "data/tfrecord/train_dataset.tfrecord"
test_tfrecord_path = "data/tfrecord/test_dataset.tfrecord"
vocab_path = "references/vocab.pickle"
n_pos_neg_path = "references/n_positive_negative.json"
model_dir = "models/"


@invoke.task
def clear(c):
    """生成されたデータを削除"""
    with open(motif_data_path, 'r') as f:
        motif_data = json.load(f)

    # data/processed/ のファイルを削除
    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        out_dir = os.path.join(processed_dir, virus)

        for protein in content['proteins']:
            out_path = os.path.join(out_dir, f'{protein}.pickle')

            if os.path.exists(out_path):
                os.remove(out_path)

        if os.path.exists(out_dir):
            os.rmdir(out_dir)

    # data/tfrecord/ のファイルを削除
    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        out_dir = os.path.join(eval_tfrecord_dir, virus)

        for protein in content['proteins']:
            out_path = os.path.join(out_dir, f'{protein}.tfrecord')

            if os.path.exists(out_path):
                os.remove(out_path)

        if os.path.exists(out_dir):
            os.rmdir(out_dir)

    if os.path.exists(train_tfrecord_path):
        os.remove(train_tfrecord_path)

    if os.path.exists(test_tfrecord_path):
        os.remove(test_tfrecord_path)

    # reference/ のファイルを削除
    if os.path.exists(vocab_path):
        os.remove(vocab_path)

    if os.path.exists(n_pos_neg_path):
        os.remove(n_pos_neg_path)

    # models/ のファイルを削除
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        invoke.run('touch {}/.gitkeep'.format(model_dir))
