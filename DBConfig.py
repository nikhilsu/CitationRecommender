import os

import pymongo


class DBConfig(object):
    @staticmethod
    def username():
        return os.environ['MONGO_USERNAME']

    @staticmethod
    def password():
        return os.environ['MONGO_PASSWORD']

    @staticmethod
    def host():
        return os.environ['MONGO_HOST']

    @staticmethod
    def port():
        return os.environ['MONGO_PORT']

    @staticmethod
    def database():
        return os.environ['MONGO_DB']

    @staticmethod
    def dataset_collection():
        return 'data'

    @staticmethod
    def in_citation_collection():
        return 'inCitations'

    @staticmethod
    def database_url():
        return 'mongodb://{}:{}@{}:{}/{}'.format(DBConfig.username(), DBConfig.password(), DBConfig.host(),
                                                 DBConfig.port(), DBConfig.database())


class MongoClient(object):
    @staticmethod
    def mongo_database():
        database_url = DBConfig.database_url()
        client = pymongo.MongoClient(database_url)
        return client[DBConfig.database()]
