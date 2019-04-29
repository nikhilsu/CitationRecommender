from random import randint

import numpy as np
from keras.layers import K, Reshape, Dot, Flatten


# Credits: https://github.com/allenai/citeomatic/blob/master/citeomatic/models/layers.py
def triplet_loss(y_true, y_pred):
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    pos = y_pred[::2]
    neg = y_pred[1::2]
    # margin is given by the difference in labels
    margin = y_true[::2] - y_true[1::2]
    delta = K.maximum(margin + neg - pos, 0)
    return K.mean(delta, axis=-1)


def l2_normalize(x):
    return K.l2_normalize(x, axis=-1)


def cosine_distance(tensor1, tensor2, dimensions, normalize):
    reshaped_input = [(Reshape((1, dimensions))(tensor1)), (Reshape((1, dimensions))(tensor2))]
    dot_product = Dot(axes=(2, 2), normalize=normalize)(reshaped_input)
    return Flatten()(dot_product)


def random_training_doc_id(num_of_samples, split):
    return randint(1, int(num_of_samples * split))


TRUE_CITATION_OFFSET = 0.3
NESTED_NEGATIVE_OFFSET = 0.2
RANDOM_NEGATIVE_OFFSET = 0.0
margin_multiplier = 1
margins_offset_dict = {
    'positive': TRUE_CITATION_OFFSET * margin_multiplier,
    'nested_neg': NESTED_NEGATIVE_OFFSET * margin_multiplier,
    'random_neg': RANDOM_NEGATIVE_OFFSET * margin_multiplier
}
CITATION_SLOPE = 0.01
MAX_CITATION_BOOST = 0.02


def compute_label(doc_in_citation_count, offset_type):
    sigmoid = 1 / (1 + np.exp(-doc_in_citation_count * CITATION_SLOPE))
    return margins_offset_dict[offset_type] + (sigmoid * MAX_CITATION_BOOST)
