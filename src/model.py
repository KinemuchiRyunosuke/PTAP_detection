import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertConfig

def build_model():
    config = BertConfig(
        num_labels=2
    )
    model = TFBertForSequenceClassification(
        config=config
    )
    model.layers[-1].activation = tf.keras.activations.sigmoid
    return model
