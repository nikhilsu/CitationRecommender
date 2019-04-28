import tensorflow as tf
from keras.engine import Model
from keras.layers import Add
from keras.optimizers import TFOptimizer

from models.word_embeddings.base_word_embeddings import BaseWordEmbeddings
from models.word_embeddings.custom_layer import LambdaScalarMultiplier
from models.word_embeddings.helpers.utils import triplet_loss, l2_normalize


class DenseWordEmbedding(object):
    def __init__(self, input_dimension, output_dimensions, learning_rate, epochs, steps_per_epoch):
        embedding = BaseWordEmbeddings(input_dimension, output_dimensions).create_model()
        self.title_embedding_multiplier = LambdaScalarMultiplier(name='title_weights')
        self.abstract_embedding_multiplier = LambdaScalarMultiplier(name='abstract_weights')
        self.title_embedding = embedding
        self.abstract_embedding = embedding
        self.model = self.__compile_model()
        optimizer = TFOptimizer(tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate))
        self.model.compile(optimizer=optimizer, loss=triplet_loss)
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

    def __compile_model(self):
        title_weights = self.title_embedding_multiplier(self.title_embedding.outputs[0])
        abstract_weights = self.abstract_embedding_multiplier(self.abstract_embedding.outputs[0])
        normalized_weighted_sum = l2_normalize(Add()([title_weights, abstract_weights]))
        model_inputs = [self.title_embedding.input, self.abstract_embedding.input]
        return Model(inputs=model_inputs, outputs=normalized_weighted_sum)

    def fit(self, dataset_generator):
        self.model.fit_generator(generator=dataset_generator,
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs)
