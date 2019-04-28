import time

from dataset_generator.raw_dataset import RawDataset
from dataset_generator.word_embeddings.embedding_triplets_generator import EmbeddingTripletsGenerator


def train():
    batch_size = 32
    dataset_generator = EmbeddingTripletsGenerator(RawDataset())
    start = time.time()
    training_data = dataset_generator.generate_triplets_for_epoch(batch_size)
    end = time.time()
    print('Generated {} triplets in {:0.2f}secs'.format(len(training_data), end - start))
