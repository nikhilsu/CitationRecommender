import pymongo

from mongo_connector.db_config import DBConfig


class MongoClient(object):
    @staticmethod
    def mongo_database():
        database_url = DBConfig.database_url()
        client = pymongo.MongoClient(database_url)
        return client[DBConfig.database()]
