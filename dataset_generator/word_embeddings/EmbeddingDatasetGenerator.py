from enum import Enum
from random import randint


class Technique(Enum):
    RANDOM = 1
    NESTED_CITE = 2


class EmbeddingDatasetGenerator(object):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset

    def __random_neg_document(self, doc_id, n=1):
        out_citations = self.raw_dataset.oout_citations(doc_id)
        exclude_list = [doc_id] + out_citations

        def random_id(excluding, range_lim, retry_count):
            if retry_count > 100:
                raise Exception('Cannot generate Random Negative Sample')
            rand_id = randint(1, range_lim)
            # Using recursion to optimize for memory while "re-rolling"
            return random_id(exclude_list, range_lim, retry_count + 1) if rand_id in excluding else rand_id

        lim = self.raw_dataset.count()
        return [self.raw_dataset.find_one_by_doc_id(random_id(exclude_list, lim, 0)) for _ in range(n)]

    def __nested_citation_neg_document(self, doc_id, n=None):
        out_citations = self.raw_dataset.out_citation_ids(doc_id)
        nested_out_citations = []
        for cite_id in out_citations:
            nested_out_citations += self.raw_dataset.out_citation_ids(cite_id)

        nested_cite_neg_docs = []
        ids = range(len(nested_out_citations)) if not n else [randint(0, len(nested_out_citations) - 1) for _ in n]
        for index in ids:
            nested_cite_neg_docs.append(self.raw_dataset.find_one_by_doc_id(nested_out_citations[index]))
        return nested_cite_neg_docs

    def negative_document(self, doc_id, technique, n):
        if technique == Technique.RANDOM:
            return self.__random_neg_document(doc_id, n)

        else:
            return self.__nested_citation_neg_document(doc_id, n)

    def positive_document(self, doc_id):
        return self.raw_dataset.out_citation_docs(doc_id)

    def training_triplets(self, doc_id, max_triplets):
        d_q = self.raw_dataset.find_one_by_doc_id(doc_id)
        d_pos = self.positive_document(doc_id)
        if len(d_pos) == 0:
            return []
        d_pos = d_pos[:max(max_triplets, len(d_pos))]
        n_rand_neg = len(d_pos) // 2
        n_nested_neg = len(d_pos) - n_rand_neg
        d_neg = self.negative_document(doc_id, Technique.RANDOM, n_rand_neg)
        d_neg += self.negative_document(doc_id, Technique.NESTED_CITE, n_nested_neg)

        # No cross-product
        return zip([d_q] * len(d_pos), d_pos, d_neg)

    def generate_training_data(self, split, max_triplets=float('-inf')):
        assert 0 < split < 1
        total = self.raw_dataset.count()
        train_split = int(total * split)
        i = 1
        train_ids = set()
        while i <= train_split:
            rand_doc_id = randint(1, total)
            if rand_doc_id not in train_ids:
                i += 1
                train_ids.add(rand_doc_id)

        return [self.training_triplets(doc_id, max_triplets) for doc_id in train_ids]

    def generate_training_data_for_epoch(self, batch_size, triplets_per_doc_id=3):
        remaining = batch_size
        train_ids = set()
        training_data = []
        while remaining > 0:
            rand_doc_id = randint(1, self.raw_dataset.count())
            if rand_doc_id not in train_ids:
                train_ids.add(rand_doc_id)
                triplets = self.training_triplets(rand_doc_id, min(remaining, triplets_per_doc_id))
                remaining -= len(triplets)
                training_data += triplets

        return training_data
