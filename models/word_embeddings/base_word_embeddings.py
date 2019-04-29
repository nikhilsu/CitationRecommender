from keras import Input, Model
from keras.layers import SpatialDropout1D, K
from keras.regularizers import l2, l1
from traitlets import Float

from models.word_embeddings.custom_layer import EmbeddingLayer
from models.word_embeddings.helpers.utils import l2_normalize


class BaseWordEmbeddings(object):
    def __init__(self, dense_dim, n_features):
        self.dense_dim = dense_dim
        self.n_features = n_features
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
        model_input = Input(shape=(None,), dtype='int32')
        direction = l2_normalize(self.direction_embedding(model_input))
        magnitude = self.scalar_magnitude(model_input)
        composite_embedding = self.dropout(direction * magnitude)

        normalized_sum = l2_normalize(K.sum(composite_embedding, axis=1))
        return Model(inputs=model_input, outputs=[normalized_sum])
