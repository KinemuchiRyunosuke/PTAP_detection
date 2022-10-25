import os
import json
import pickle
import optuna
import tensorflow as tf

from dataset import make_dataset
from fit_vocab import fit_vocab
from preprocessing import preprocess_dataset, write_tfrecord
from train_model import train
from evaluate import evaluate_precision
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

# TEST===========================================================
# batch_size = 100000
# epochs = 1
# hopping_num = 1
# hidden_dim = 8
# ===============================================================

# paths
motif_data_path = 'references/PTAP_data.json'
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
    study.optimize(objective, n_trials=50)

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
    separate_len = trial.suggest_int('separate_len', 2, 4)
    seq_length = length - separate_len + 2
    num_words = num_amino_acid ** separate_len

    # データセット作成
    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        out_dir = os.path.join(processed_dir.format(separate_len), virus)
        dataset = make_dataset(
                motif_data=motif_data,
                length=length,
                virus=virus,
                fasta_dir=fasta_dir,
                separate_len=separate_len,
                n_gram=n_gram)

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

    fit_vocab(motif_data=motif_data,
              num_words=num_words,
              dataset_dir=processed_dir.format(separate_len),
              vocab_path=vocab_path)

    # データセットの前処理
    x_train, x_test, y_train, y_test = \
        preprocess_dataset(
            motif_data=motif_data,
            processed_dir=processed_dir.format(separate_len),
            eval_tfrecord_dir=eval_tfrecord_dir.format(separate_len),
            vocab_path=vocab_path,
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

    with open(motif_data_path, 'r') as f:
        motif_data = json.load(f)

    precision = evaluate_precision(
            motif_data=motif_data,
            model=model,
            seq_length=seq_length,
            batch_size=batch_size,
            threshold=threshold,
            eval_tfrecord_dir=eval_tfrecord_dir.format(separate_len))
    return -precision

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


if __name__ == '__main__':
    main()
