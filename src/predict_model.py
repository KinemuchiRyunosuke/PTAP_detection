import os
import json
import pickle
import tensorflow as tf

from dataset import make_dataset
from preprocessing import preprocess_dataset, Vocab
from train_model import train
from evaluate import evaluate
from model import build_model


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

# TEST===========================================================
batch_size = 100
epochs = 1
hopping_num = 1
hidden_dim = 8
# ===============================================================

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
            out_dir = os.path.join(processed_dir, virus)
            dataset = make_dataset(
                    motif_data=motif_data,
                    length=length,
                    virus=virus,
                    fasta_dir=fasta_dir)

            # TEST======================================================
            for key, (x, y) in dataset.items():
                dataset[key] = (x[:1000], y[:1000])
            #===========================================================

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # データセットを保存
            for protein, (x, y) in dataset.items():
                out_path = os.path.join(out_dir, f'{protein}.pickle')
                with open(out_path, 'wb') as f:
                    pickle.dump(x, f)
                    pickle.dump(y, f)

    vocab = Vocab(separate_len=separate_len, class_token=True)

    if not (os.path.exists(train_tfrecord_path) \
            and os.path.exists(test_tfrecord_path)):
        print("================== PREPROCESSING ===================")

        # データセットの前処理
        preprocess_dataset(
            motif_data=motif_data,
            processed_dir=processed_dir,
            train_tfrecord_path=train_tfrecord_path,
            test_tfrecord_path=test_tfrecord_path,
            eval_tfrecord_dir=eval_tfrecord_dir,
            vocab=vocab,
            n_pos_neg_path=n_pos_neg_path,
            val_rate=val_rate,
            seed=seed)


    model = build_model(
            hidden_size=hidden_dim,
            vocab_size=num_words + num_tokens,
            num_attention_heads=head_num,
            num_hidden_layers=hopping_num,
            attention_probs_dropout_prob=dropout_rate
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(
                                learning_rate=lr),
                 loss="sparse_categorical_crossentropy")

    if not os.path.exists(checkpoint_path):
        print("================== TRAINING ========================")
        model = train(model=model,
                      seq_length=length - separate_len + 2,
                      batch_size=batch_size,
                      epochs=epochs,
                      n_pos_neg_path=n_pos_neg_path,
                      train_tfrecord_path=train_tfrecord_path,
                      test_tfrecord_path=test_tfrecord_path,
                      checkpoint_path=model_dir)

    print("================== EVALUATION ======================")
    evaluate(motif_data=motif_data,
             model=model,
             seq_length=length - separate_len + 2,
             batch_size=batch_size,
             threshold=threshold,
             eval_tfrecord_dir=eval_tfrecord_dir,
             vocab=vocab,
             result_path=result_path,
             false_positive_path=false_positive_path,
             positive_pred_path=positive_pred_path)


def finish_making_dataset(motif_data):
    finish = True

    for content in motif_data:
        virus = content['virus'].replace(' ', '_')
        dataset_dir = os.path.join(processed_dir, virus)
        for protein in content['proteins']:
            dataset_path = os.path.join(dataset_dir, f'{protein}.pickle')

            if not os.path.exists(dataset_path):
                finish = False

    return finish


if __name__ == '__main__':
    main()
