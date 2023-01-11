import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertConfig

def build_model(hidden_size, vocab_size, num_attention_heads,
        num_hidden_layers, attention_probs_dropout_prob):
    config = BertConfig(
        num_labels=2,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        attention_probs_dropout_prob=attention_probs_dropout_prob,


    )
    model = TFBertForSequenceClassification(
        config=config
    )
    model.layers[-1].activation = tf.keras.activations.sigmoid
    return model
