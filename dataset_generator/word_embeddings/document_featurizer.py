import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


class DocumentFeaturizer(object):
    STOPWORDS = {
        'abstract', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
        'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the',
        'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with',
        'the', 'we', 'our', 'which'
    }

    def __init__(self, raw_dataset, train_split, max_abstract_len, max_title_len):
        self.raw_dataset = raw_dataset
        self.max_abstract_len = max_abstract_len
        self.max_title_len = max_title_len

        title_abstract_of_training_data = self.raw_dataset.fetch_collated_training_text(train_split)
        max_df_frac = 0.90
        min_df_frac = 0.000025
        self.count_vectorizer = CountVectorizer(
            max_df=max_df_frac,
            min_df=min_df_frac,
            stop_words=self.STOPWORDS
        )
        self.count_vectorizer.fit(tqdm(title_abstract_of_training_data))
        self.word_to_index = dict((word, index + 1) for index, word in enumerate(self.count_vectorizer.vocabulary_))

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
        common_types = [np.intersect1d(d_qs, candidate) for (d_qs, candidate) in zip(d_qs, candidates)]
        common_types_features = np.zeros_like(d_qs)
        for i, intersection in enumerate(common_types):
            common_types_features[i, :len(intersection)] = intersection
        return common_types_features

    def featurize_documents(self, documents):
        features = {
            'title':
                np.asarray([self.__extract_textual_features(doc['title'], self.max_title_len) for doc in documents]),
            'abstract':
                np.asarray(
                    [self.__extract_textual_features(doc['abstract'], self.max_abstract_len) for doc in documents])
        }

        return features

    def features_of_triplet(self, train_examples):
        d_qs, candidates = train_examples

        d_q_features = self.featurize_documents(d_qs)
        candidate_features = self.featurize_documents(candidates)
        citation_features = DocumentFeaturizer.__extract_citation_features(candidates)
        common_title_features = DocumentFeaturizer.__extract_common_types_features(d_q_features['title'],
                                                                                   candidate_features['title'])
        common_abstract_features = DocumentFeaturizer.__extract_common_types_features(d_q_features['abstract'],
                                                                                      candidate_features['abstract'])

        return {
            'query-title-text':
                d_q_features['title'],
            'query-abstract-text':
                d_q_features['abstract'],
            'candidate-title-text':
                candidate_features['title'],
            'candidate-abstract-text':
                candidate_features['abstract'],
            'query-candidate-common-title':
                common_title_features,
            'query-candidate-common-abstract':
                common_abstract_features,
            'candidate-citation-count':
                citation_features
        }
