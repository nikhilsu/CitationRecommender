from random import randint

from mongo_connector.db_config import DBConfig
from mongo_connector.mongo_client import MongoClient


class RawDataset(object):
    def __init__(self, database=MongoClient.mongo_database()):
        self.database = database
        self.__num_records = None

    def __collection(self, collection_name):
        return self.database[collection_name]

    def __add_in_citation_count_to_doc(self, doc):
        in_count = self.__collection(DBConfig.dataset_collection()).find({'out_citations': str(doc['id'])}).count()
        doc['in_citation_count'] = in_count

    # inCitations - List of paper IDs which cited this(doc_id) paper.
    def in_citation_ids(self, doc_id):
        if DBConfig.in_citation_collection() in self.database.list_collection_names():
            return self.__collection(DBConfig.in_citation_collection()).find_one({'id': str(doc_id)})['citedBy']
        return map(lambda rec: rec['id'],
                   self.__collection(DBConfig.dataset_collection()).find({'out_citations': str(doc_id)}))

    # outCitations - List of paper IDs which this paper(doc_id) cited.
    def out_citation_ids(self, doc_id):
        return self.find_one_by_doc_id(doc_id)['out_citations']

    def out_citation_docs(self, doc_id, max_docs):
        out_citation_ids = self.out_citation_ids(doc_id)
        return out_citation_ids, self.find_by_doc_ids(out_citation_ids[:min(max_docs, len(out_citation_ids))], True)

    def find_one_by_doc_id(self, doc_id):
        doc = self.__collection(DBConfig.dataset_collection()).find_one({'id': str(doc_id)})
        self.__add_in_citation_count_to_doc(doc)
        return doc

    def find_by_doc_ids(self, doc_ids, post_processing=False):
        output = []
        docs = self.__collection(DBConfig.dataset_collection()).find({'id': {'$in': [str(d_id) for d_id in doc_ids]}})
        if not post_processing:
            return docs
        for doc in docs:
            self.__add_in_citation_count_to_doc(doc)
            output.append(doc)
        return output

    def count(self):
        if self.__num_records is None:
            self.__num_records = self.__collection(DBConfig.dataset_collection()).count()
        return self.__num_records

    def fetch_collated_training_text(self, train_split):
        return list(map(lambda doc: ' '.join((doc['title'], doc['abstract'])),
                        self.__collection(DBConfig.dataset_collection()).find().limit(int(self.count() * train_split))))

    def all_documents_generator(self):
        return self.__collection(DBConfig.dataset_collection()).find()

    def fetch_random_document(self):
        rand_doc_id = randint(1, self.count())
        return self.find_one_by_doc_id(rand_doc_id)
