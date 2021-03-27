import os
from random import shuffle
import gensim
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import re
from os.path import join
from gensim.models.doc2vec import Doc2Vec
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import remove_stopwords
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import LSTM
from keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
import logging
import multiprocessing
import tensorflow as tf
import collections

from csv import reader
import pandas as pd
from io import StringIO

from nltk.corpus import reuters
import nltk
nltk.download('reuters')
nltk.download('punkt')

from Doc2Vec import D2V
from data_storer import DataStorer


# import tensorflow.keras as keras

class MovieDataset:

    def __init__(self):
        self.reader = FileReader()
        self.model_movie = D2V()
        self.data_handler = DataStorer()
        self.model_name = "movie_d2v.d2v"
        self.train_topics = list()
        self.train_docs = list()
        self.test_topics = list()
        self.test_docs = list()
        self.doc_labels = None

        self.labels = []

        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def load_d2v(self, q, param):
        if not os.path.exists(os.path.join(self.reader.get_models_path(), self.model_name)):
            q.put([1, "No D2V models found. Please Train a D2V model"])
        else:
            q.put([0, "Model Found. Loading it in now."])
            self.model_movie.load(self.reader.get_models_path(), self.model_name)
            self.load_data()
            self.labels = self.data_handler.get_topics()

            # print(self.doc_labels)
            # print(self.labels)
            if param == 0:
                q.put([0, "D2V model loaded Successfully"])
            else:
                q.put([1, "D2V model loaded Successfully"])

    def load_data(self):
        self.doc_labels = self.model_movie.get_labels()
        for label in self.doc_labels:
            self.data_handler.add_document(self.model_movie.get_doc_vec(label))
            split_string = label.split("__")
            self.data_handler.add_topic(split_string[0])

    def train_d2v(self, q):
        q.put([0, "Staring training of Doc2Vec model"])
        train_corpus = list(self.reader.read_corpus_train(q))
        # print("FINISHED HERE")
        result = self.model_movie.train(train_corpus)
        # print(result)
        if result == 1:
            q.put([0, "Doc2Vec Training Complete! Attempting to save D2V model now."])
            self.model_movie.save(self.reader.get_models_path(), self.model_name)
            q.put([1, "D2v Model Saved."])

    def train_dataset(self, q):

        self.load_d2v(q, 0)
        self.load_trainset(q)
        self.load_testset(q)

        self.X_train = np.array(self.train_docs)
        self.Y_train = np.array(self.test_docs)
        self.X_test = np.array(self.train_topics)
        self.Y_test = np.array(self.test_topics)

        self.X_train = np.reshape(self.X_train, (len(self.X_train), 300, 1))
        self.X_test = np.reshape(self.X_test, (len(self.X_test), 2, 1))
        self.Y_train = np.reshape(self.Y_train, (len(self.Y_train), 300, 1))
        self.Y_test = np.reshape(self.Y_test, (len(self.Y_test), 2, 1))

        self.X_test = tf.squeeze(self.X_test, axis=-1)
        self.Y_test = tf.squeeze(self.Y_test, axis=-1)

        # print(self.X_train.shape)
        # print(self.X_test.shape)
        # print(self.Y_train.shape)
        # print(self.Y_test.shape)

        q.put([0, "Begining Model Training"])

        model_training = Sequential()
        model_training.add(InputLayer(input_shape=(300, 1)))
        model_training.add(Conv1D(filters=32, kernel_size=8, padding='same', activation='relu'))
        model_training.add(Dropout(0.5))
        model_training.add(Dense(512))
        model_training.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
        model_training.add(Dropout(0.25))
        model_training.add(Dense(256))
        model_training.add(Dense(128))
        model_training.add(MaxPooling1D(pool_size=3))
        model_training.add(Flatten())
        model_training.add(Dense(2))
        model_training.add(Activation('sigmoid'))
        model_training.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model_training.summary())

        model_training.fit(self.X_train, self.X_test, validation_data=(self.Y_train, self.Y_test), batch_size=128,
                           epochs=10)
        model_training.save("G:/FYP_Work/CNN_models/movie_review_model")
        # model_training.save("C:/Users/eogha/Documents/Workspace/CNN_Models/movie_review_model")
        # Final evaluation of the model
        scores = model_training.evaluate(self.Y_train, self.Y_test, verbose=0)
        accuracy = scores[1] * 100
        q.put([1, "Accuracy %s " % accuracy])

    def test_dataset(self, q):
        self.load_d2v(q, 0)
        # Relative Paths. Not absolute paths.****
        model_training = tf.keras.models.load_model("G:/FYP_Work/CNN_models/movie_review_model")
        # model_training = tf.keras.models.load_model("C:/Users/eogha/Documents/Workspace/CNN_Models/movie_review_model")
        self.load_testset(q)

        self.Y_train = np.array(self.test_docs)
        self.Y_test = np.array(self.test_topics)

        self.Y_train = np.reshape(self.Y_train, (len(self.Y_train), 300, 1))
        self.Y_test = np.reshape(self.Y_test, (len(self.Y_test), 2, 1))
        self.Y_test = tf.squeeze(self.Y_test, axis=-1)

        loss, acc = model_training.evaluate(self.Y_train, self.Y_test, batch_size=128)
        result = "Test Data Accuracy: " + "{:.1%}".format(acc)
        q.put([1, result])

    def test_dataset_new_input(self, q, text_input):

        config = tf.compat.v1.ConfigProto(gpu_options=
                                          tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                          # device_count = {'GPU': 1}
                                          )
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)

        self.model_movie.load(self.reader.get_models_path(), self.model_name)
        model_training = tf.keras.models.load_model("G:/FYP_Work/CNN_models/movie_review_model")
        # model_training = tf.keras.models.load_model("C:/Users/eogha/Documents/Workspace/CNN_Models/movie_review_model")
        processed_content = simple_preprocess(remove_stopwords(text_input))
        new_doc2vec = np.array(self.model_movie.infer_doc(processed_content))
        new_doc2vec = np.reshape(new_doc2vec, (1, 300, 1))
        prediction = (model_training.predict(new_doc2vec))
        # print(prediction)
        temp = []
        for val in prediction[0]:
            temp.append(round(val, 2))

        result = "Bad Review    : " + "{:.1%}".format((temp[0])) + "\n"
        result += "Good Review : " + "{:.1%}".format((temp[1]))

        q.put([1, result])

    def get_topic_vector(self, t):
        topic_vec = list()
        for topic in self.labels:
            if t == topic:
                topic_vec.append(1)
            else:
                topic_vec.append(0)

        return topic_vec

    def load_trainset(self, q):

        # print("Loading Training Set")
        q.put([0, "Loading Training Set"])
        self.doc_labels = self.model_movie.get_labels()
        # print(self.doc_labels)
        self.train_topics.clear()
        self.train_docs.clear()

        for label in self.doc_labels:
            # print("Current label: %s" % label)
            self.train_docs.append(self.model_movie.get_doc_vec(label))
            split_string = label.split("__")

            self.train_topics.append(split_string[0])
            if split_string[0] not in self.labels:
                self.labels.append(split_string[0])

        for i in range(len(self.train_topics)):
            self.train_topics[i] = self.get_topic_vector(self.train_topics[i])

        q.put([0, "Finished loading Training set"])

    def load_testset(self, q):
        self.test_topics.clear()
        self.test_docs.clear()

        # print("Loading test dataset")
        q.put([0, "Loading Test Set"])
        topics = self.labels
        # print(topics)

        for topic in topics:
            # print("Current topic: %s" % topic)
            q.put([0, "Current topic: %s" % topic])
            file_location = os.path.join(self.reader.get_testing_path(), topic)
            files = os.listdir(file_location)

            for file in files:
                with open(os.path.join(file_location, file), mode="r", encoding="utf-8") as file:
                    content = file.read()
                cleaned_doc = simple_preprocess(remove_stopwords(content))

                self.test_topics.append(topic)
                self.test_docs.append(self.model_movie.infer_doc(cleaned_doc))

        for i in range(len(self.test_topics)):
            self.test_topics[i] = self.get_topic_vector(self.test_topics[i])

        # print("Finished loading test set")
        q.put([0, "Finished loading test set"])


class ReutersDataset:

    def __init__(self):
        self.model_reuters = D2V()
        self.model_name = "reuters_d2v.d2v"
        self.doc2vec_model_location = "G:/FYP_Work/doc2vec_models"
        self.model_reuters_path = "G:/FYP_Work/CNN_models/reuters_model"
        self.doc2vec_dimensions = 300

    def load_d2v(self, q, param):

        if not os.path.exists(os.path.join(self.doc2vec_model_location, self.model_name)):
            q.put([1, "No D2V models found. Please Train a D2V model"])
        else:
            q.put([0, "Model Found. Loading it in now."])
            self.model_reuters.load(self.doc2vec_model_location, self.model_name)
            if param == 0:
                q.put([0, "D2V model loaded Successfully"])
            else:
                q.put([1, "D2V model loaded Successfully"])

    def train_d2v(self, q):
        q.put([0, "Staring training of Doc2Vec model"])
        taggedDocuments = [
            TaggedDocument(words=gensim.utils.simple_preprocess(remove_stopwords(reuters.raw(fileId).lower())),
                           tags=[i]) for i, fileId in enumerate(reuters.fileids())]

        result = self.model_reuters.train(taggedDocuments)
        if result == 1:
            q.put([0, "Doc2Vec Training Complete! Attempting to save D2V model now."])
            self.model_reuters.save(self.doc2vec_model_location, self.model_name)
            q.put([1, "D2v Model Saved."])

    def train_dataset(self, q):
        self.load_d2v(q, 0)
        train_articles = [{'raw': reuters.raw(fileId), 'categories': reuters.categories(fileId)} for fileId in
                          reuters.fileids() if fileId.startswith('training/')]
        test_articles = [{'raw': reuters.raw(fileId), 'categories': reuters.categories(fileId)} for fileId in
                         reuters.fileids() if fileId.startswith('test/')]

        labelBinarizer = MultiLabelBinarizer()
        labelBinarizer.fit([reuters.categories(fileId) for fileId in reuters.fileids()])

        train_data = [self.model_reuters.infer_doc(gensim.utils.simple_preprocess(remove_stopwords(article['raw'].lower()))) for
                      article in train_articles]
        # print("train_data Complete")
        q.put([0, "train_data Complete"])
        test_data = [self.model_reuters.infer_doc(gensim.utils.simple_preprocess(remove_stopwords(article['raw'].lower()))) for article
                     in test_articles]
        # print("test_data Complete")
        q.put([0, "test_data Complete"])
        train_labels = labelBinarizer.transform([article['categories'] for article in train_articles])
        # print("train_labels Complete")
        q.put([0, "train_labels Complete"])
        test_labels = labelBinarizer.transform([article['categories'] for article in test_articles])
        # print("test_labels Complete")
        q.put([0, "test_labels Complete"])
        train_data, test_data, train_labels, test_labels = np.asarray(train_data), np.asarray(test_data), np.asarray(
            train_labels), np.asarray(test_labels)

        train_data = np.reshape(train_data, (len(train_data), 300, 1))
        train_labels = np.reshape(train_labels, (len(train_labels), 90, 1))
        test_data = np.reshape(test_data, (len(test_data), 300, 1))
        test_labels = np.reshape(test_labels, (len(test_labels), 90, 1))

        print(train_data)

        train_labels = tf.squeeze(train_labels, axis=-1)
        test_labels = tf.squeeze(test_labels, axis=-1)

        model_training = Sequential()
        model_training.add(InputLayer(input_shape=(300, 1)))
        model_training.add(Conv1D(filters=32, kernel_size=8, padding='same', activation='relu'))
        model_training.add(Dropout(0.5))
        model_training.add(Dense(512))
        model_training.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
        model_training.add(Dropout(0.25))
        model_training.add(Dense(256))
        model_training.add(Dense(128))
        model_training.add(MaxPooling1D(pool_size=3))
        model_training.add(Flatten())
        model_training.add(Dense(train_labels.shape[1]))
        model_training.add(Activation('sigmoid'))
        model_training.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model_training.summary())

        model_training.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=128,
                           epochs=50)

        model_training.save(self.model_reuters_path)

        # Final evaluation of the model
        loss, acc = model_training.evaluate(test_data, test_labels, verbose=0)
        result = "Accuracy: " + "{:.1%}".format(acc)
        q.put([1, result])

    def test_dataset(self, q):

        config = tf.compat.v1.ConfigProto(gpu_options=
                                          tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                          # device_count = {'GPU': 1}
                                          )
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)

        self.load_d2v(q, 0)
        model_training = tf.keras.models.load_model(self.model_reuters_path)

        test_articles = [{'raw': reuters.raw(fileId), 'categories': reuters.categories(fileId)} for fileId in
                         reuters.fileids() if fileId.startswith('test/')]
        test_data = [self.model_reuters.infer_doc(gensim.utils.simple_preprocess(remove_stopwords(article['raw']))) for article in test_articles]

        test_data = np.reshape(test_data, (len(test_data), 300, 1))

        predictions = model_training.predict(np.asarray(test_data))

        predictions[predictions < 0.5] = 0
        predictions[predictions >= 0.5] = 1

        labelBinarizer = MultiLabelBinarizer()
        labelBinarizer.fit([reuters.categories(fileId) for fileId in reuters.fileids()])
        predicted_labels = labelBinarizer.inverse_transform(predictions)

        # print(len(predicted_labels))
        # result = ""
        for predicted_label, test_article in zip(predicted_labels, test_articles):
            # print('title: {}'.format(test_article['raw'].splitlines()[0])) + "\n"
            # print('predicted: {} - actual: {}'.format(list(predicted_label), test_article['categories']))
            result = 'title: {}'.format(test_article['raw'].splitlines()[0])
            q.put([0, result])
            result = 'predicted: {} - actual: {}'.format(list(predicted_label), test_article['categories'])
            q.put([0, result])
        # print(result)
        # q.put([0, result])

        test_data = [self.model_reuters.infer_doc(gensim.utils.simple_preprocess(remove_stopwords(article['raw']))) for article in test_articles]
        test_labels = labelBinarizer.transform([article['categories'] for article in test_articles])
        test_data = np.reshape(test_data, (len(test_data), 300, 1))
        test_labels = np.reshape(test_labels, (len(test_labels), 90, 1))

        loss, acc = model_training.evaluate(test_data, test_labels, batch_size=128)

        accResult = "Accuracy: " + "{:.1%}".format(acc)
        q.put([1, accResult])


class FileReader:
    def __init__(self):

        # self.__testing_path = "C:/Users/eogha/Documents/Workspace/FYP_Datasets/Large_Movie_dataset/aclImdb/test"
        # self.__training_path = "C:/Users/eogha/Documents/Workspace/FYP_Datasets/Large_Movie_dataset/aclImdb/train"
        # self.__models_paths = "C:/Users/eogha/Documents/Workspace/doc2vec_models"
        self.__models_paths = "G:/FYP_Work/doc2vec_models"
        self.__training_path = "G:/FYP_Work/FYP_Datasets/Large_Movie_dataset/aclImdb/train"
        self.__testing_path = "G:/FYP_Work/FYP_Datasets/Large_Movie_dataset/aclImdb/test"

    def read_corpus_train(self, q):

        folders = dirs = os.listdir(self.__training_path)

        # Go through each folder in the training dataset.
        for folder in dirs:
            q.put([0, "Current folder: {}".format(folder)])
            # print("Current folder: {}".format(folder))
            curr_path = os.path.join(self.__training_path, folder)
            docs = os.listdir(curr_path)
            for i, document in enumerate(docs):
                curr_doc_write = os.path.join(curr_path, document)
                with open(curr_doc_write, mode="r", encoding="utf-8") as file:
                    content = file.read()
                    doc_id = folder + "__" + str(i)
                    yield gensim.models.doc2vec.TaggedDocument(
                        gensim.utils.simple_preprocess(remove_stopwords(content)), [doc_id])

    def get_models_path(self):
        return self.__models_paths

    def get_training_path(self):
        return self.__training_path

    def get_testing_path(self):
        return self.__testing_path
