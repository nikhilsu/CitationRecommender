import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


class DocumentFeaturizer(object):
    STOPWORDS = {
        'abstract', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
        'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the',
        'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with',
        'the', 'we', 'our', 'which'
    }

    def __init__(self, raw_dataset, opts):
        self.raw_dataset = raw_dataset
        self.max_abstract_len = opts.max_abstract_len
        self.max_title_len = opts.max_title_len

        title_abstract_of_training_data = self.raw_dataset.fetch_collated_training_text(opts.train_split)
        max_df_frac = 0.90
        self.count_vectorizer = CountVectorizer(
            max_df=max_df_frac,
            max_features=opts.max_features,
            stop_words=self.STOPWORDS
        )
        self.count_vectorizer.fit(tqdm(title_abstract_of_training_data, desc='Building Count-Vectorizer'))
        self.word_to_index = dict((word, index + 1) for index, word in enumerate(self.count_vectorizer.vocabulary_))
        self.n_features = 1 + len(self.word_to_index)
        opts.n_features = self.n_features

    def __index_of_word(self, word):
        return self.word_to_index[word] if word in self.word_to_index else None

    def __word_to_index_features(self, document):
        x_indexes = []
        for words in document:
            indexes = []
            for word in words:
                index = self.__index_of_word(word)
                if index:
                    indexes.append(index)
            x_indexes.append(indexes)
        return x_indexes

    def __extract_textual_features(self, text, max_len):
        return np.asarray(pad_sequences(self.__word_to_index_features([text]), max_len)[0], dtype=np.int32)

    @staticmethod
    def __extract_citation_features(documents):
        return np.log([max(doc['in_citation_count'] - 1, 0) + 1 for doc in documents])

    @staticmethod
    def __extract_common_types_features(d_qs, candidates):
        common_types = [np.intersect1d(d_q, candidate) for (d_q, candidate) in zip(d_qs, candidates)]
        common_types_features = np.zeros_like(d_qs)
        for i, intersection in enumerate(common_types):
            common_types_features[i, :len(intersection)] = intersection
        return common_types_features

    @staticmethod
    def __extract_sim_scores(d_qs, candidates, candidate_selector):
        return np.asarray(
            [candidate_selector.cosine_similarity(d_q, candidate) for (d_q, candidate) in zip(d_qs, candidates)])

    def featurize_documents(self, documents):
        features = {
            'title':
                np.asarray([self.__extract_textual_features(doc['title'], self.max_title_len) for doc in documents]),
            'abstract':
                np.asarray(
                    [self.__extract_textual_features(doc['abstract'], self.max_abstract_len) for doc in documents])
        }

        return features

    def extract_features(self, d_qs, candidates, candidate_selector=None):
        for_nn_rank = candidate_selector is not None
        d_q_features = self.featurize_documents(d_qs)
        candidate_features = self.featurize_documents(candidates)
        features = {
            'query-title-text':
                d_q_features['title'],
            'query-abstract-text':
                d_q_features['abstract'],
            'candidate-title-text':
                candidate_features['title'],
            'candidate-abstract-text':
                candidate_features['abstract']
        }
        if for_nn_rank:
            citation_features = DocumentFeaturizer.__extract_citation_features(candidates)
            common_title = DocumentFeaturizer.__extract_common_types_features(d_q_features['title'],
                                                                              candidate_features['title'])
            common_abstract = DocumentFeaturizer.__extract_common_types_features(d_q_features['abstract'],
                                                                                 candidate_features['abstract'])
            similarity_score_features = DocumentFeaturizer.__extract_sim_scores(d_qs, candidates, candidate_selector)

            features['query-candidate-common-title'] = common_title
            features['query-candidate-common-abstract'] = common_abstract
            features['candidate-citation-count'] = citation_features
            features['similarity-score'] = similarity_score_features
        return features
