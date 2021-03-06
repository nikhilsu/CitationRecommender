from random import randint

from models.word_embeddings.helpers.utils import compute_label, random_training_doc_id


class EmbeddingTripletsGenerator(object):
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
        return self.raw_dataset.find_by_doc_ids(random_ids, True)

    def __nested_citation_neg_document(self, out_citation_ids, n=None):
        nested_out_citations = []
        for cite_id in out_citation_ids:
            nested_out_citations += self.raw_dataset.out_citation_ids(cite_id)

        if len(nested_out_citations) == 0:
            return []

        ids = range(len(nested_out_citations)) if not n else set([randint(0, len(nested_out_citations) - 1) for _ in
                                                                  range(n)])
        return self.raw_dataset.find_by_doc_ids([nested_out_citations[index] for index in ids], True)

    def __positive_document(self, doc_id, max_docs):
        return self.raw_dataset.out_citation_docs(doc_id, max_docs)

    def __triplets(self, doc_id, required_num_samples):
        d_q = self.raw_dataset.find_one_by_doc_id(doc_id)
        n_pos = n_nested_neg = ((3 * required_num_samples) // 10)
        out_citation_ids, d_pos = self.__positive_document(doc_id, n_pos)

        d_nested_neg = self.__nested_citation_neg_document(out_citation_ids, n_nested_neg) if len(d_pos) != 0 else []
        exclude_list = set()
        exclude_list.add(int(doc_id))
        exclude_list.update([int(cite_id) for cite_id in out_citation_ids])
        for document in d_nested_neg:
            exclude_list.add(int(document['id']))

        n_rand_neg = required_num_samples - (len(d_pos) + len(d_nested_neg))
        d_rand_neg = self.__random_neg_document(exclude_list, n_rand_neg)

        pos_labels = [compute_label(doc['in_citation_count'], 'positive') for doc in d_pos]

        neg_labels = [compute_label(doc['in_citation_count'], 'nested_neg') for doc in d_nested_neg] + \
                     [compute_label(doc['in_citation_count'], 'random_neg') for doc in d_rand_neg]
        total = len(d_pos) + len(d_nested_neg) + len(d_rand_neg)
        return d_q, [d_pos, d_nested_neg + d_rand_neg], [pos_labels, neg_labels], total

    def generate_triplets_for_epoch(self, batch_size, train_split, triplets_per_doc_id=4):
        remaining = batch_size
        triplet_ids = set()
        x_d_qs = []
        x_candidates = []
        x_labels = []
        while remaining > 0:
            rand_doc_id = random_training_doc_id(self.raw_dataset.count(), train_split)
            if rand_doc_id not in triplet_ids:
                triplet_ids.add(rand_doc_id)
                output = self.__triplets(rand_doc_id, min(remaining, triplets_per_doc_id))
                if len(output) == 0:
                    continue
                d_q, candidates, labels, read = output
                remaining -= read
                x_d_qs += [d_q] * read
                x_candidates += candidates[0]
                x_candidates += candidates[1]
                x_labels += labels[0]
                x_labels += labels[1]
        return x_d_qs, x_candidates, x_labels
