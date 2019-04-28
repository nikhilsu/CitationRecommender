# Credits: https://github.com/allenai/citeomatic/blob/master/citeomatic/models/layers.py

from keras import Input, Model
from keras.layers import Embedding, K, SpatialDropout1D
from keras.regularizers import l1, l2
from traitlets import Float


class EmbeddingLayer(Embedding):
    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        out = K.gather(self.embeddings, inputs)
        mask = K.expand_dims(K.clip(K.cast(inputs, 'float32'), 0, 1), axis=-1)
        return out * mask


class DocumentWordEmbeddings(object):
    def __init__(self, input_dimension, output_dimension):
        self.dense_dim = input_dimension
        self.n_features = output_dimension
        self.l1_lambda = Float(default_value=0.0000001)
        self.l2_lambda = Float(default_value=0.00001)
        self.dropout_p = Float(default_value=0)

        self.direction_embedding = EmbeddingLayer(
            output_dim=self.dense_dim,
            input_dim=self.n_features,
            activity_regularizer=l2(self.l2_lambda),
        )

        self.scalar_magnitude = EmbeddingLayer(
            output_dim=1,
            input_dim=self.n_features,
            activity_regularizer=l1(self.l1_lambda),
            embeddings_initializer='uniform'
        )

        self.dropout = SpatialDropout1D(self.dropout_p)

    def create_model(self):
        direction = K.l2_normalize(self.direction_embedding(Input(shape=(None,), dtype='int32')), axis=-1)
        magnitude = self.scalar_magnitude(Input(shape=(None,), dtype='int32'))
        composite_embedding = self.dropout(direction * magnitude)
        model_input = (Input(shape=(None,), dtype='int32'))

        normalized_sum = K.l2_normalize(K.sum(composite_embedding, axis=1))
        return Model(inputs=model_input, outputs=[normalized_sum]), [normalized_sum]
