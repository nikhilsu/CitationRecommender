from dataset_generator.raw_dataset import RawDataset
from dataset_generator.word_embeddings.document_featurizer import DocumentFeaturizer
from dataset_generator.word_embeddings.embedding_triplets_generator import EmbeddingTripletsGenerator
from dataset_generator.word_embeddings.features_generator import EmbeddingFeaturesGenerator
from models.nn_rank.nn_rank import NNRank
from models.nn_select.candidate_selector import CandidateSelector
from models.word_embeddings.dense_embedding_model import DenseEmbeddingModel


def train(opts):
    raw_dataset = RawDataset()
    dataset_generator = EmbeddingTripletsGenerator(raw_dataset)
    print('Creating vocabulary')
    featurizer = DocumentFeaturizer(raw_dataset, opts)
    features_generator = EmbeddingFeaturesGenerator(dataset_generator, featurizer, opts.batch_size, opts.train_split)

    print('Built Count Vectorizer Vocabulary')
    dense_model = DenseEmbeddingModel(featurizer, opts)
    dense_model.load_embedding_model_weights(opts.embedding_model_weights_path)
    dense_model.load_dense_model_weights(opts.dense_model_weights_path)

    candidate_selector = CandidateSelector(raw_dataset, dense_model, opts.knn)
    model = NNRank(dense_model.nn_rank_model, opts)
    print('Starting training of model')
    model.fit(features_generator.yield_features_generator(candidate_selector))