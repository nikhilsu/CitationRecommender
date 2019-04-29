from keras import Input, Model
from keras.layers import SpatialDropout1D
from keras.regularizers import l2, l1

from models.word_embeddings.custom_layer import EmbeddingLayer
from models.word_embeddings.helpers.utils import l2_normalize_layer, summation_layer, product_layer


class BaseWordEmbeddings(object):
    def __init__(self, dense_dim, n_features, l1_lambda, l2_lambda, dropout_p):
        self.dense_dim = dense_dim
        self.n_features = n_features
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.dropout_p = dropout_p

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

        self.dropout_layer = SpatialDropout1D(self.dropout_p)

    def create_model(self, name_prefix):
        model_input = Input(shape=(None,), dtype='int32', name='{}-text'.format(name_prefix))
        direction_embeddings = self.direction_embedding(model_input)
        magnitude_embedding = self.scalar_magnitude(model_input)
        normalized_direction_embeddings = l2_normalize_layer()(direction_embeddings)
        product_embedding = product_layer()([normalized_direction_embeddings, magnitude_embedding])
        composite_embedding = self.dropout_layer(product_embedding)

        summation_composite_embeddings = summation_layer()(composite_embedding)
        normalized_sum = l2_normalize_layer()(summation_composite_embeddings)
        return Model(inputs=model_input, outputs=[normalized_sum], name='{}-embedding-model'.format(name_prefix))
