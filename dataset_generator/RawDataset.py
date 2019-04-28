from mongo_connector.DBConfig import DBConfig
from mongo_connector.MongoClient import MongoClient


class RawDataset(object):
    def __init__(self, database=MongoClient.mongo_database()):
        self.database = database
        self.num_records = None

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
        return self.find_one_by_doc_id(doc_id)['out_citations']

    def in_citation_docs(self, doc_id):
        in_citation_ids = self.in_citation_ids(doc_id)
        return self.find_by_doc_ids(in_citation_ids)

    def out_citation_docs(self, doc_id, max_docs):
        out_citation_ids = self.out_citation_ids(doc_id)
        return out_citation_ids, self.find_by_doc_ids(out_citation_ids[:min(max_docs, len(out_citation_ids))])

    def find_one_by_doc_id(self, doc_id):
        return self.__collection(DBConfig.dataset_collection()).find_one({'id': str(doc_id)})

    def find_by_doc_ids(self, doc_ids):
        return [doc for doc in
                self.__collection(DBConfig.dataset_collection()).find({'id': {'$in': [str(d_id) for d_id in doc_ids]}})]

    def count(self):
        if self.num_records is None:
            self.num_records = self.__collection(DBConfig.dataset_collection()).count()
        return self.num_records
