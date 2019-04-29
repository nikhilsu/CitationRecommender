import os

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Add
from keras.optimizers import TFOptimizer

from models.word_embeddings.base_word_embeddings import BaseWordEmbeddings
from models.word_embeddings.callbacks.save_model_weights import SaveModelWeights
from models.word_embeddings.custom_layer import LambdaScalarMultiplier
from models.word_embeddings.helpers.utils import triplet_loss, cosine_distance, l2_normalize_layer


class DenseEmbeddingModel(object):
    def __init__(self, featurizer, opts):
        self.featurizer = featurizer
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
        self.model, self.embedding_model = self.__compile_dense_model()
        optimizer = TFOptimizer(tf.contrib.opt.LazyAdamOptimizer(learning_rate=opts.learning_rate))
        self.model.compile(optimizer=optimizer, loss=triplet_loss)
        self.nn_rank_model = dict()

        self.callbacks = [
            SaveModelWeights(self.model, self.embedding_model, opts.weights_directory, opts.checkpoint_frequency)
        ]

    def __compile_embedding_model(self, document_name):
        title_model = self.title_embedding.create_model('{}-{}'.format(document_name, 'title'))
        abstract_model = self.abstract_embedding.create_model('{}-{}'.format(document_name, 'abstract'))
        self.nn_rank_model['{}-title'.format(document_name)] = title_model
        self.nn_rank_model['{}-abstract'.format(document_name)] = abstract_model

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
        return Model(inputs=model_inputs, outputs=model_output), Model(inputs=query_input, outputs=query_model_output)

    def fit(self, dataset_generator):
        self.model.fit_generator(generator=dataset_generator,
                                 steps_per_epoch=self.steps_per_epoch,
                                 callbacks=self.callbacks,
                                 epochs=self.epochs)

    def predict_dense_embeddings(self, documents, pre_trained_weights_path=None):
        if pre_trained_weights_path is not None:
            self.load_embedding_model_weights(pre_trained_weights_path)

        document_embeddings = []
        for doc in documents:
            features = self.featurizer.featurize_documents([doc])
            doc_embedding = self.embedding_model.predict(
                {
                    'query-title-text': features['title'],
                    'query-abstract-text': features['abstract'],
                    'document-txt': features['abstract'],
                }
            )
            document_embeddings.append(doc_embedding)
        return np.asarray(document_embeddings)

    def save_weights(self, directory, only_embedding_model=True):
        self.embedding_model.save_weights(os.path.join(directory, 'embedding_model_weights.h5'))
        if not only_embedding_model:
            self.model.save_weights(os.path.join(directory, 'composite_embedding_model.h5'))

    def load_composite_model_weights(self, path):
        self.model.load_weights(path)
        print('Loaded Composite Weights')

    def load_embedding_model_weights(self, path):
        self.embedding_model.load_weights(path)
        print('Loaded Dense Embedding Weights')
