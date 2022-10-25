import json
import tensorflow as tf

from preprocessing import load_dataset


def train(model, seq_length, batch_size, epochs, n_pos_neg_path,
          train_tfrecord_path, test_tfrecord_path):
    # クラス重みを設定
    with open(n_pos_neg_path, 'r') as f:
        n_pos_neg = json.load(f)

    total = n_pos_neg['n_positive'] + n_pos_neg['n_negative']
    positive_weight = (1/n_pos_neg['n_positive']) * total / 2.0
    negative_weight = (1/n_pos_neg['n_negative']) * total / 2.0
    class_weight = {0: positive_weight, 1: negative_weight}

    train_ds = load_dataset(train_tfrecord_path,
                            batch_size=batch_size,
                            length=seq_length)
    test_ds = load_dataset(test_tfrecord_path,
                            batch_size=batch_size,
                            length=seq_length)

    # 学習
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
                monitor='val_precision', mode='max', patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_precision', mode='max',
                factor=0.2, patience=3)
    ]
    model.fit(x=train_ds,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=test_ds,
            callbacks=callbacks,
            shuffle=True,
            class_weight=class_weight,
            verbose=1)
