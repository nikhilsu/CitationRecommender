from DBConfig import DBConfig
from DBConfig import MongoClient


# inCitations  List of S2 paper IDs which cited this paper.
# outCitations  List of S2 paper IDs which this paper cited.


class RawDataset(object):
    def __init__(self, database=MongoClient.mongo_database()):
        self.database = database

    def __collection(self, collection_name):
        return self.database[collection_name]

    def in_citations(self, doc_id):
        if DBConfig.in_citation_collection() in self.database.list_collection_names():
            return self.__collection(DBConfig.in_citation_collection()).find_one({'id': str(doc_id)})['citedBy']
        return map(lambda rec: rec['id'],
                   self.__collection(DBConfig.dataset_collection()).find({'out_citations': str(doc_id)}))

    def out_citations(self, doc_id):
        return self.__collection(DBConfig.dataset_collection()).find_one({'id': str(doc_id)})['out_citations']

    def find_one_by_doc_id(self, doc_id):
        return self.__collection(DBConfig.dataset_collection()).find_one({'id': str(doc_id)})

    def count(self, collection_name):
        if collection_name in self.database.list_collection_names():
            return self.__collection(collection_name).count()
        return -1
