

# DataStorer class used for the Movie Review Dataset
class DataStorer:

    def __init__(self):
        self.__topics = list()
        self.__loaded_topics = list()
        self.__documents = list()

    def add_topic(self, t: str):
        if t not in self.__topics:
            self.__topics.append(t)

    def get_topics(self):
        return self.__topics

    def add_document(self, doc):
        self.__documents.append(doc)

    def clear_topics(self):
        self.__topics.clear()

    def clear_documents(self):
        self.__documents.clear()

    def clear(self):
        self.__topics.clear()
        self.__documents.clear()
        self.__loaded_topics.clear()
