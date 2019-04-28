from DBConfig import DBConfig
from DBConfig import MongoClient


class RawDataset(object):
    def __init__(self, database=MongoClient.mongo_database()):
        self.database = database

    def __collection(self, collection_name):
        return self.database[collection_name]

    # inCitations - List of paper IDs which cited this(doc_id) paper.
    def in_citation_ids(self, doc_id):
        if DBConfig.in_citation_collection() in self.database.list_collection_names():
            return self.__collection(DBConfig.in_citation_collection()).find_one({'id': str(doc_id)})['citedBy']
        return map(lambda rec: rec['id'],
                   self.__collection(DBConfig.dataset_collection()).find({'out_citations': str(doc_id)}))

    # outCitations - List of paper IDs which this paper(doc_id) cited.
    def out_citation_ids(self, doc_id):
        return self.__collection(DBConfig.dataset_collection()).find_one({'id': str(doc_id)})['out_citations']

    def in_citation_docs(self, doc_id):
        in_citation_ids = self.in_citation_ids(doc_id)
        return self.find_by_doc_ids(in_citation_ids)

    def out_citation_docs(self, doc_id):
        out_citation_ids = self.out_citation_ids(doc_id)
        return self.find_by_doc_ids(out_citation_ids)

    def find_one_by_doc_id(self, doc_id):
        return self.__collection(DBConfig.dataset_collection()).find_one({'id': str(doc_id)})

    def find_by_doc_ids(self, doc_ids):
        return [self.find_one_by_doc_id(doc_id) for doc_id in doc_ids]

    def count(self, collection_name):
        if collection_name in self.database.list_collection_names():
            return self.__collection(collection_name).count()
        return -1
