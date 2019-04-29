import tensorflow as tf
from keras import Model
from keras.layers import Add
from keras.optimizers import TFOptimizer

from models.word_embeddings.base_word_embeddings import BaseWordEmbeddings
from models.word_embeddings.custom_layer import LambdaScalarMultiplier
from models.word_embeddings.helpers.utils import triplet_loss, cosine_distance, l2_normalize_layer


class DenseWordEmbedding(object):
    def __init__(self, opts):
        self.dropout_p = opts.dropout_p
        self.l2_lambda = opts.l2_lambda
        self.l1_lambda = opts.l1_lambda
        self.n_features = opts.n_features
        self.dense_dims = opts.dense_dims
        self.epochs = opts.epochs
        self.steps_per_epoch = opts.steps_per_epoch

        embeddings = BaseWordEmbeddings(self.dense_dims, self.n_features, self.l1_lambda, self.l2_lambda,
                                        self.dropout_p)
        self.title_embedding = embeddings
        self.abstract_embedding = embeddings
        self.title_embedding_multiplier = LambdaScalarMultiplier(name='title-scalar-multiplier')
        self.abstract_embedding_multiplier = LambdaScalarMultiplier(name='abstract-scalar-multiplier')
        self.model = self.__compile_dense_model()
        optimizer = TFOptimizer(tf.contrib.opt.LazyAdamOptimizer(learning_rate=opts.learning_rate))
        self.model.compile(optimizer=optimizer, loss=triplet_loss)

    def __compile_embedding_model(self, document_name):
        title_model = self.title_embedding.create_model('{}-{}'.format(document_name, 'title'))
        abstract_model = self.abstract_embedding.create_model('{}-{}'.format(document_name, 'abstract'))
        title_weights = self.title_embedding_multiplier(title_model.outputs[0])
        abstract_weights = self.abstract_embedding_multiplier(abstract_model.outputs[0])

        sum_weights = Add()([title_weights, abstract_weights])
        normalized_weighted_sum = l2_normalize_layer()(sum_weights)

        model_inputs = [title_model.input, abstract_model.input]
        return model_inputs, normalized_weighted_sum

    def __compile_dense_model(self):
        query_input, query_model_output = self.__compile_embedding_model('query')
        candidate_input, candidate_model_output = self.__compile_embedding_model('candidate')
        model_output = cosine_distance(query_model_output, candidate_model_output, self.dense_dims, False)
        model_inputs = query_input + candidate_input
        return Model(inputs=model_inputs, outputs=model_output)

    def fit(self, dataset_generator):
        self.model.fit_generator(generator=dataset_generator,
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs)
