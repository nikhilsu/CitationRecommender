import numpy as np


class EmbeddingFeaturesGenerator(object):
    def __init__(self, triplet_generator, featurizer, batch_size, train_split):
        self.train_split = train_split
        self.featurizer = featurizer
        self.triplet_generator = triplet_generator
        self.batch_size = batch_size

    def yield_features_generator(self, candidate_selector=None):
        while True:
            triplet = self.triplet_generator.generate_triplets_for_epoch(self.batch_size, self.train_split)
            queries, candidates, labels = triplet
            yield (self.featurizer.extract_features(queries, candidates, candidate_selector), np.asarray(labels))
