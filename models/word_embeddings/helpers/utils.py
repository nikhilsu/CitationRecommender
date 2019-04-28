from keras.layers import K


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
