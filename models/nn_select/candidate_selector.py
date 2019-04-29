import numpy as np
from keras.layers import K
from tqdm import tqdm

from dataset_generator.raw_dataset import RawDataset
from models.word_embeddings.dense_embedding_model import DenseEmbeddingModel


class CandidateSelector(object):
    def __init__(self, raw_dataset: RawDataset, dense_embedding_model: DenseEmbeddingModel, knn):
        self.raw_dataset = raw_dataset
        self.dense_embedding_model = dense_embedding_model
        self.__create_dense_embedding_space()
        self.knn = knn

    def __create_dense_embedding_space(self):
        embeddings_output_shape = K.int_shape(self.dense_embedding_model.embedding_model.outputs[0])[-1]
        self.embedding_space = np.zeros((self.raw_dataset.count(), embeddings_output_shape))

        for document in tqdm(self.raw_dataset.all_documents_generator()):
            word_embeddings = self.dense_embedding_model.predict_dense_embeddings([document])
            doc_id = int(document['id'])
            self.embedding_space[doc_id - 1] = word_embeddings

    def nearest_neighbors_of(self, doc_embedding):
        cosine_similarities = np.dot(self.embedding_space, -doc_embedding)
        knn_embeddings = np.argpartition(cosine_similarities, self.knn)[:self.knn]
        doc_ids = np.argsort(cosine_similarities[knn_embeddings]) + 1
        return doc_ids

    def cosine_similarity(self, query_embedding, candidate_ids):
        return np.dot(self.embedding_space[candidate_ids - 1], query_embedding)

    def fetch_candidates_with_similarities(self, query_document):
        query_embedding = self.dense_embedding_model.predict_dense_embeddings([query_document])
        candidate_ids = self.nearest_neighbors_of(query_embedding)
        return zip(candidate_ids, self.cosine_similarity(query_embedding, candidate_ids))
