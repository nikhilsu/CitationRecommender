from dataset_generator.raw_dataset import RawDataset
from dataset_generator.word_embeddings.document_featurizer import DocumentFeaturizer
from dataset_generator.word_embeddings.embedding_triplets_generator import EmbeddingTripletsGenerator
from dataset_generator.word_embeddings.features_generator import EmbeddingFeaturesGenerator
from models.word_embeddings.dense_embedding_model import DenseWordEmbedding


def train(opts):
    raw_dataset = RawDataset()
    dataset_generator = EmbeddingTripletsGenerator(raw_dataset)
    featurizer = DocumentFeaturizer(raw_dataset, opts)
    features_generator = EmbeddingFeaturesGenerator(dataset_generator, featurizer, opts.batch_size, opts.train_split)

    opts.n_features = featurizer.n_features
    model = DenseWordEmbedding(opts)
    model.fit(features_generator.yield_features_generator())
