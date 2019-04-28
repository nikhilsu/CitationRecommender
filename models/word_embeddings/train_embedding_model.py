import time

from dataset_generator.RawDataset import RawDataset
from dataset_generator.word_embeddings.EmbeddingDatasetGenerator import EmbeddingDatasetGenerator


def train():
    batch_size = 32
    dataset_generator = EmbeddingDatasetGenerator(RawDataset())
    start = time.time()
    training_data = dataset_generator.generate_training_data_for_epoch(batch_size)
    end = time.time()
    print('Generated {} triplets in {:0.2f}secs'.format(len(training_data), end - start))
