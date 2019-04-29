from keras import Input
from keras.regularizers import l1

from models.word_embeddings.custom_layer import EmbeddingLayer
from models.word_embeddings.helpers.utils import cosine_distance, summation_layer


class NNRank(object):
    def __init__(self, nn_rank_embedding_model, dense_dims, opts):
        self.opts = opts
        self.dense_dims = dense_dims
        self.query_text_model = nn_rank_embedding_model['query-text']
        self.query_abstract_model = nn_rank_embedding_model['query-abstract']
        self.candidate_text_model = nn_rank_embedding_model['candidate-text']
        self.candidate_abstract_model = nn_rank_embedding_model['candidate-abstract']

        model_inputs = [self.query_text_model.input, self.candidate_text_model.input,
                        self.query_abstract_model.input, self.candidate_abstract_model]

        pre_dense_network_output = []

        cos_sim_text = cosine_distance(self.query_text_model.output[0], self.candidate_text_model.output[0],
                                       self.dense_dims, True)

        cos_sim_abstract = cosine_distance(self.query_abstract_model.output[0], self.candidate_abstract_model.output[0],
                                           self.dense_dims, True)

        pre_dense_network_output.append(cos_sim_text)
        pre_dense_network_output.append(cos_sim_abstract)

        for field in ['text', 'abstract']:
            common_type_input = Input(
                name='query-candidate-common-{}'.format(field), shape=(None,)
            )
            elementwise_sparse = EmbeddingLayer(
                input_dim=self.opts.n_features,
                output_dim=1,
                mask_zero=True,
                name="%s-sparse-embedding" % field,
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
