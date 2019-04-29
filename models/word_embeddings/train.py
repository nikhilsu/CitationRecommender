import time

from dataset_generator.raw_dataset import RawDataset
from dataset_generator.word_embeddings.embedding_triplets_generator import EmbeddingTripletsGenerator
from dataset_generator.word_embeddings.features_generator import EmbeddingFeaturesGenerator


def train(opts):
    dataset_generator = EmbeddingTripletsGenerator(RawDataset())
    features_generator = EmbeddingFeaturesGenerator(dataset_generator, opts.batch_size)
    start = time.time()
    training_data = dataset_generator.generate_triplets_for_epoch(opts.batch_size, opts.train_split)
    end = time.time()
    print('Generated {} triplets in {:0.2f}secs'.format(len(training_data), end - start))
