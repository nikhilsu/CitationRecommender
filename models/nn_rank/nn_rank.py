from keras import Input, Model
from keras.layers import Concatenate, Dense
from keras.regularizers import l1

from models.word_embeddings.callbacks.save_model_weights import SaveModelWeights
from models.word_embeddings.custom_layer import EmbeddingLayer
from models.word_embeddings.helpers.utils import cosine_distance, summation_layer, triplet_loss


class NNRank(object):
    def __init__(self, nn_rank_embedding_model, opts):
        self.opts = opts
        self.dense_dims = opts.dense_dims
        self.query_title_model = nn_rank_embedding_model['query-title']
        self.query_abstract_model = nn_rank_embedding_model['query-abstract']
        self.candidate_title_model = nn_rank_embedding_model['candidate-title']
        self.candidate_abstract_model = nn_rank_embedding_model['candidate-abstract']

        model_inputs = [self.query_title_model.input, self.candidate_title_model.input,
                        self.query_abstract_model.input, self.candidate_abstract_model.input]

        pre_dense_network_output = []

        cos_sim_text = cosine_distance(self.query_title_model.output, self.candidate_title_model.output,
                                       self.dense_dims, True)

        cos_sim_abstract = cosine_distance(self.query_abstract_model.output, self.candidate_abstract_model.output,
                                           self.dense_dims, True)

        pre_dense_network_output.append(cos_sim_text)
        pre_dense_network_output.append(cos_sim_abstract)

        for field in ['title', 'abstract']:
            common_type_input = Input(
                name='query-candidate-common-{}'.format(field), shape=(None,)
            )
            elementwise_sparse = EmbeddingLayer(
                input_dim=self.opts.n_features,
                output_dim=1,
                mask_zero=True,
                name="{}-sparse-embedding".format(field),
                activity_regularizer=l1(self.opts.l1_lambda)
            )(common_type_input)
            pre_dense_network_output.append(summation_layer()(elementwise_sparse))
            model_inputs.append(common_type_input)

        citation_count_input = Input(shape=(1,), dtype='float32', name='candidate-citation-count')
        model_inputs.append(citation_count_input)
        pre_dense_network_output.append(citation_count_input)

        similarity_score_input = Input(shape=(1,), dtype='float32', name='similarity-score')
        model_inputs.append(similarity_score_input)
        pre_dense_network_output.append(similarity_score_input)

        input_to_dense_layer = Concatenate()(pre_dense_network_output)
        output_dense_one = Dense(20, name='dense-1', activation='elu')(input_to_dense_layer)
        output_dense_two = Dense(20, name='dense-2', activation='elu')(output_dense_one)
        nn_rank_output = Dense(1, kernel_initializer='one', name='final-output', activation='sigmoid')(output_dense_two)
        self.model = Model(inputs=model_inputs, outputs=nn_rank_output)
        self.model.compile(optimizer='nadam', loss=triplet_loss)

        self.callbacks = [SaveModelWeights([('nn_rank', self.model)], self.opts.weights_directory,
                                           self.opts.checkpoint_frequency)]

    def fit(self, dataset_generator):
        self.model.fit_generator(generator=dataset_generator,
                                 steps_per_epoch=self.opts.steps_per_epoch,
                                 callbacks=self.callbacks,
                                 epochs=self.opts.epochs)
