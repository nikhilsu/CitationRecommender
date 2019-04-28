from enum import Enum
from random import randint

from DBConfig import DBConfig


class Technique(Enum):
    RANDOM = 1
    NESTED_CITE = 2


class EmbeddingDatasetGenerator(object):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset

    def __random_neg_document(self, doc_id):
        in_citations = self.raw_dataset.in_citations(doc_id)
        exclude_list = [doc_id] + in_citations

        def random_id(excluding, range_lim, retry_count):
            if retry_count > 100:
                raise Exception('Cannot generate Random Negative Sample')
            rand_id = randint(1, range_lim)
            return random_id(exclude_list, range_lim, retry_count + 1) if rand_id in excluding else rand_id

        lim = self.raw_dataset.count(DBConfig.dataset_collection())
        neg_doc_id = random_id(exclude_list, lim, 0)
        return self.raw_dataset.find_one_by_doc_id(neg_doc_id)

    def negative_document(self, doc_id, technique):
        if technique == Technique.RANDOM:
            return self.__random_neg_document(doc_id)
