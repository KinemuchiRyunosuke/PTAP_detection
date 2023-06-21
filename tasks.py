import os
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
reports_dir = 'reports/'


@invoke.task
def clear(c):
    """生成されたデータを削除"""

    # data/processed/ のファイルを削除
    rmdir(processed_dir, git_keep=True)

    # data/tfrecord/ のファイルを削除
    rmdir(eval_tfrecord_dir, git_keep=True)
    remove(train_tfrecord_path)
    remove(test_tfrecord_path)

    # reference/ のファイルを削除
    remove(vocab_path)
    remove(n_pos_neg_path)

    # models/ のファイルを削除
    rmdir(model_dir, git_keep=True)

    # reports/ のファイルを削除
    rmdir(reports_dir, git_keep=True)

def remove(path):
    if os.path.exists(path):
        os.remove(path)

def rmdir(dir_path, git_keep=True):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

        if git_keep:
            invoke.run('touch {}/.gitkeep'.format(dir_path))
