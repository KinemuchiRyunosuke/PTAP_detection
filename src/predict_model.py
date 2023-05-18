import os
import json
import pickle
import tensorflow as tf

from dataset import make_dataset
from fit_vocab import fit_vocab
from preprocessing import preprocess_dataset, write_tfrecord, Vocab
from train_model import train
from evaluate import evaluate
from models.transformer import BinaryClassificationTransformer


test_mode = True

# parameters
length = 26
n_gram = True
val_rate = 0.2
num_amino_acid = 20
separate_len = 1
num_words = num_amino_acid ** separate_len
num_tokens = 2
batch_size = 1024
epochs = 50
threshold = 0.5         # 陽性・陰性の閾値
head_num = 8            # Transformerの並列化に関するパラメータ
dropout_rate = 0.04
hopping_num = 3         # Multi-Head Attentionを施す回数
hidden_dim = 184        # 単語ベクトルの次元数
lr = 2.60e-5            # 学習率
beta = 0.5              # Fベータスコアの引数
seed = 1                # データセットをシャッフルするときのseed値

if test_mode:
    batch_size = 100000
    epochs = 1
    hopping_num = 1
    hidden_dim = 8

# paths
motif_data_path = 'references/motif_data.json'
fasta_dir = "data/interim/"
processed_dir = "data/processed/"
tfrecord_dir = "data/tfrecord/"
eval_tfrecord_dir = "data/tfrecord/eval/"
train_tfrecord_path = "data/tfrecord/train_dataset.tfrecord"
test_tfrecord_path = "data/tfrecord/test_dataset.tfrecord"
vocab_path = "references/vocab.pickle"
n_pos_neg_path = "references/n_positive_negative.json"
model_dir = "models/"
checkpoint_path = "models/saved_model.pb"
result_path = "reports/result/evaluation.csv"
false_positive_path = "reports/result/false_positive.csv"
positive_pred_path = "reports/result/positive_pred.csv"

def main():
    gpus = tf.config.list_physical_devices(device_type='GPU')
    if len(gpus) > 0:
        print(f">> GPU detected. {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)

    with open(motif_data_path, 'r') as f:
        motif_data = json.load(f)

    if not finish_making_dataset(motif_data):
        print("================== MAKING DATASET ==================")
        # データセットを生成
        for content in motif_data:
            virus = content['virus'].replace(' ', '_')
            xs, ys = make_dataset(
                    motif_data=motif_data,
                    length=length,
                    virus=virus,
                    fasta_dir=fasta_dir,
                    separate_len=separate_len)

            if test_mode:
                xs = xs[:10000]
                ys = ys[:10000]

            # データセットを保存
            out_path = os.path.join(processed_dir, f'{virus}.pickle')
            with open(out_path, 'wb') as f:
                pickle.dump(xs, f)
                pickle.dump(ys, f)

    if not os.path.exists(vocab_path):
        print("================== FITTING =========================")
        vocab = fit_vocab(motif_data=motif_data,
                          num_words=num_words,
                          dataset_dir=processed_dir)

        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab.tokenizer, f)

    else:
        with open(vocab_path, 'rb') as f:
            tokenizer = pickle.load(f)
        vocab = Vocab(tokenizer)


    if not (os.path.exists(train_tfrecord_path) \
            and os.path.exists(test_tfrecord_path)):
        print("================== PREPROCESSING ===================")

        # データセットの前処理
        x_train, x_test, y_train, y_test = \
            preprocess_dataset(
                motif_data=motif_data,
                processed_dir=processed_dir,
                eval_tfrecord_dir=eval_tfrecord_dir,
                vocab=vocab,
                n_pos_neg_path=n_pos_neg_path,
                val_rate=val_rate,
                seed=seed)

        # tf.data.Datasetとして保存
        write_tfrecord(x_test, y_test, test_tfrecord_path)
        write_tfrecord(x_train, y_train, train_tfrecord_path)

    model = create_model()

    if not os.path.exists(checkpoint_path):
        print("================== TRAINING ========================")
        train(model=model,
              seq_length=length - separate_len + 2,
              batch_size=batch_size,
              epochs=epochs,
              n_pos_neg_path=n_pos_neg_path,
              train_tfrecord_path=train_tfrecord_path,
              test_tfrecord_path=test_tfrecord_path,
              checkpoint_path=model_dir)

    print("================== EVALUATION ======================")
    model.load_weights(os.path.dirname(checkpoint_path))
    evaluate(motif_data=motif_data,
             model=model,
             seq_length=length - separate_len + 2,
             batch_size=batch_size,
             threshold=threshold,
             eval_tfrecord_dir=eval_tfrecord_dir,
             vocab_path=vocab_path,
             result_path=result_path,
             false_positive_path=false_positive_path,
             positive_pred_path=positive_pred_path)


def finish_making_dataset(motif_data):
    finish = True

    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        dataset_path = os.path.join(processed_dir, f'{virus}.pickle')

        if not os.path.exists(dataset_path):
            finish = False

    return finish


def create_model():
    """ モデルを定義する """
    model = BinaryClassificationTransformer(
                vocab_size=num_words + num_tokens,
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
