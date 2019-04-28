from dataset_generator.RawDataset import RawDataset
from dataset_generator.word_embeddings.EmbeddingDatasetGenerator import EmbeddingDatasetGenerator

batch_size = 32

dataset_generator = EmbeddingDatasetGenerator(RawDataset())

output = dataset_generator.generate_training_data_for_epoch(32)

print(len(output))
