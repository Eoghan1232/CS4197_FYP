from os.path import join
from gensim.models.doc2vec import Doc2Vec
import multiprocessing


# Doc2Vec class Functions: train the Doc2Vec model, save the model, load the model, infer a document, get the vector
# value for a document
class D2V:

    def __init__(self):
        cores = multiprocessing.cpu_count()
        self.__model = Doc2Vec(dm=0,
                               vector_size=300,
                               min_count=2,
                               epochs=70,
                               workers=cores - 1)

    # Method used to train the Doc2Vec model. Takes the train corpus as input. returns 1 when trained successfully.
    def train(self, train_corpus):
        self.__model.build_vocab(train_corpus)
        self.__model.train(train_corpus, total_examples=self.__model.corpus_count, epochs=self.__model.epochs)
        return 1

    def save(self, folder_path, filename):
        self.__model.save(join(folder_path, filename))

    def load(self, folder_path, filename):
        self.__model = Doc2Vec.load(join(folder_path, filename))

    def infer_doc(self, doc):
        return self.__model.infer_vector(doc)

    def get_doc_vec(self, identifier: str):
        return self.__model.docvecs[identifier]

    def get_labels(self):
        return list(self.__model.docvecs.doctags.keys())
