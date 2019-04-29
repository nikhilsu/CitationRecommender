class EmbeddingFeaturesGenerator(object):
    def __init__(self, triplet_generator, batch_size, featurizer):
        self.featurizer = featurizer
        self.triplet_generator = triplet_generator
        self.batch_size = batch_size

    def create_generator(self):
        for triplet in self.triplet_generator.generate_triplets_for_epoch(self.batch_size, 0.85):
            pass
