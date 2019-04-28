from random import randint


class EmbeddingDatasetGenerator(object):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset

    def __random_neg_document(self, exclude_list, n=1):
        def random_id(excluding, range_lim, retry_count):
            if retry_count > 100:
                raise Exception('Cannot generate Random Negative Sample')
            rand_id = randint(1, range_lim)
            # Using recursion to optimize for memory while "re-rolling"
            return random_id(exclude_list, range_lim, retry_count + 1) if rand_id in excluding else rand_id

        lim = self.raw_dataset.count()
        random_ids = [random_id(exclude_list, lim, 0) for _ in range(n)]
        return self.raw_dataset.find_by_doc_ids(random_ids)

    def __nested_citation_neg_document(self, out_citation_ids, n=None):
        nested_out_citations = []
        for cite_id in out_citation_ids:
            nested_out_citations += self.raw_dataset.out_citation_ids(cite_id)

        ids = range(len(nested_out_citations)) if not n else [randint(0, len(nested_out_citations) - 1) for _ in
                                                              range(n)]
        return self.raw_dataset.find_by_doc_ids([nested_out_citations[index] for index in ids])

    def positive_document(self, doc_id, max_docs):
        return self.raw_dataset.out_citation_docs(doc_id, max_docs)

    def training_triplets(self, doc_id, max_triplets):
        d_q = self.raw_dataset.find_one_by_doc_id(doc_id)
        out_citation_ids, d_pos = self.positive_document(doc_id, max_triplets)
        if len(d_pos) == 0:
            return []

        n_rand_neg = len(d_pos) // 2
        n_nested_neg = len(d_pos) - n_rand_neg

        d_neg = self.__nested_citation_neg_document(out_citation_ids, n_nested_neg)
        exclude_list = [doc_id] + out_citation_ids + [document['id'] for document in d_neg]
        d_neg += self.__random_neg_document(exclude_list, n_rand_neg)

        min_len = min(len(d_pos), len(d_neg))
        if min_len == 0:
            return []

        # No cross-product
        return list(zip([d_q] * min_len, d_pos[:min_len], d_neg[:min_len]))

    def generate_training_data(self, split, max_triplets=float('inf')):
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
