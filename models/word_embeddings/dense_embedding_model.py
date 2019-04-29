import tensorflow as tf
from keras import Model
from keras.layers import Add
from keras.optimizers import TFOptimizer

from models.word_embeddings.base_word_embeddings import BaseWordEmbeddings
from models.word_embeddings.custom_layer import LambdaScalarMultiplier
from models.word_embeddings.helpers.utils import triplet_loss, l2_normalize, cosine_distance


class DenseWordEmbedding(object):
    def __init__(self, opts):
        self.dense_dims = opts.dense_dims
        embedding = BaseWordEmbeddings(opts.dense_dims, opts.n_features, opts.l1_lambda, opts.l2_lambda,
                                       opts.dropout_p).create_model()
        self.title_embedding_multiplier = LambdaScalarMultiplier(name='title_weights')
        self.abstract_embedding_multiplier = LambdaScalarMultiplier(name='abstract_weights')
        self.title_embedding = embedding
        self.abstract_embedding = embedding
        self.model = self.__compile_dense_model()
        optimizer = TFOptimizer(tf.contrib.opt.LazyAdamOptimizer(learning_rate=opts.learning_rate))
        self.model.compile(optimizer=optimizer, loss=triplet_loss)
        self.epochs = opts.epochs
        self.steps_per_epoch = opts.steps_per_epoch

    def __compile_embedding_model(self):
        title_weights = self.title_embedding_multiplier(self.title_embedding.outputs[0])
        abstract_weights = self.abstract_embedding_multiplier(self.abstract_embedding.outputs[0])
        normalized_weighted_sum = l2_normalize(Add()([title_weights, abstract_weights]))
        model_inputs = [self.title_embedding.input, self.abstract_embedding.input]
        return model_inputs, normalized_weighted_sum

    def __compile_dense_model(self):
        query_input, query_model_output = self.__compile_embedding_model()
        candidate_input, candidate_model_output = self.__compile_embedding_model()
        model_output = cosine_distance(query_model_output, candidate_model_output, self.dense_dims, False)
        model_inputs = query_input + candidate_input
        return Model(inputs=model_inputs, outputs=model_output)

    def fit(self, dataset_generator):
        self.model.fit_generator(generator=dataset_generator,
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs)
