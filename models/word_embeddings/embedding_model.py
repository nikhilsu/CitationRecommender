from keras.engine import Model
from keras.layers import K, Add

from models.word_embeddings.base_word_embeddings import BaseWordEmbeddings
from models.word_embeddings.custom_layer.lambda_multiplier import LambdaScalarMultiplier


class DenseWordEmbedding(object):
    def __init__(self, input_dimension, output_dimensions):
        embedding = BaseWordEmbeddings(input_dimension, output_dimensions).create_model()
        self.title_embedding_multiplier = LambdaScalarMultiplier(name='title_weights')
        self.abstract_embedding_multiplier = LambdaScalarMultiplier(name='abstract_weights')
        self.title_embedding = embedding
        self.abstract_embedding = embedding

    def create_model(self):
        title_weights = self.title_embedding_multiplier(self.title_embedding.outputs[0])
        abstract_weights = self.abstract_embedding_multiplier(self.abstract_embedding.outputs[0])
        normalized_weighted_sum = K.l2_normalize(Add()([title_weights, abstract_weights]), axis=-1)
        model_inputs = [self.title_embedding.input, self.abstract_embedding.input]
        return Model(inputs=model_inputs, outputs=normalized_weighted_sum)
