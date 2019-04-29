from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


class DocumentFeaturizer(object):
    STOPWORDS = {
        'abstract', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
        'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the',
        'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with',
        'the', 'we', 'our', 'which'
    }

    def __init__(self, raw_dataset, train_split):
        self.raw_dataset = raw_dataset
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

    def index_of_word(self, word):
        return self.word_to_index[word] if word in self.word_to_index else None

    def word_to_index_features(self, words):
        indexes = []
        for word in words:
            index = self.index_of_word(word)
            if index:
                indexes.append(index)
        return indexes
