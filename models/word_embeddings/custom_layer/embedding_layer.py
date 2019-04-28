from keras.layers import Embedding, K


# Credits: https://github.com/allenai/citeomatic/blob/master/citeomatic/models/layers.py
class EmbeddingLayer(Embedding):
    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        out = K.gather(self.embeddings, inputs)
        mask = K.expand_dims(K.clip(K.cast(inputs, 'float32'), 0, 1), axis=-1)
        return out * mask
