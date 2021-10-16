
# from ipynb.fs.full.LOAD_MODEL import Model
import tensorflow as tf
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import Model, Sequential
from tensorflow.keras.models import load_model
import numpy
import gensim
from gensim.models import KeyedVectors
import os


# #loading the word2vec models globally
# cwd_gn = os.getcwd()
# print(cwd_gn)
# path_gn = cwd_gn+"\GoogleNews-vectors-negative300.bin.gz"
# #path_gn = cwd_gn+"\w2v_100_SO.bin"
# print(path_gn)
# w2gn_model = gensim.models.KeyedVectors.load_word2vec_format(path_gn, binary = True,unicode_errors='ignore')
#
# cwd_so = os.getcwd()
# print(cwd_so)
# path_so = cwd_so+"\w2v_100_SO.bin"
# print(path_so)
# w2so_model = gensim.models.KeyedVectors.load_word2vec_format(path_so, binary = True,unicode_errors='ignore')


class Model:

    """Class responsible for loading labels, cleaning input text, tokenizing and
    identifying the sentiment of input sentences.

    """

    def load_df_labels(self):
        """
        Loads class labels from numpy file.

        Returns:
            encoded class labels

        """
        self.df_entire = pd.read_csv("SO_Dataset.csv", sep=";")
        print(self.df_entire)
        encoder = LabelEncoder()
        encoder.classes_ = numpy.load('classes.npy')
        # self.training_sentences = self.df_entire['Sentence'].to_list()
        self.training_labels = self.df_entire['Label'].to_list()
        self.label = np.array(encoder.fit_transform(self.training_labels))
        print(self.label)
        print(self.df_entire.head(10))
        return self.df_entire, self.label

    def remove_pattern(self, input_txt, pattern):
        """

        Args:
            input_txt:
            pattern:

        Returns:

        """
        r = re.findall(pattern, input_txt)
        self.input_txt = input_txt
        for i in r:
            self.input_txt = re.sub(i, '', self.input_txt)
        return self.input_txt

    def decontract(self, text):
        """

        Args:
            text:

        Returns:

        """
        self.text = re.sub(r"won\'t", "will not", text)
        self.text = re.sub(r"can\'t", "can not", self.text)
        self.text = re.sub(r"n\'t", " not", self.text)
        self.text = re.sub(r"\'re", " are", self.text)
        self.text = re.sub(r"\'s", " is", self.text)
        self.text = re.sub(r"\'m", " am", self.text)
        self.text = re.sub(r"\'d", " would", self.text)
        self.text = re.sub(r"\'ll", " will", self.text)
        self.text = re.sub(r"\'t", " not", self.text)
        self.text = re.sub("1st", "first", self.text)
        self.text = re.sub("2nd", "second", self.text)
        self.text = re.sub("3rd", "third", self.text)
        self.text = re.sub(r"\'ve", " have", self.text)
        self.text = re.sub(r"\'m", " am", self.text)
        self.text = re.sub(r"isn\'t", "is not", self.text)
        self.text = re.sub(r"doesn\'t", "does not", self.text)
        self.text = re.sub(r"didn\'t", "did not", self.text)
        self.text = re.sub(r"wasn\'t", "was not", self.text)
        self.text = re.sub(r"shouldn\'t", "should not", self.text)
        self.text = re.sub(r"hasn\'t", "has not", self.text)
        return self.text

    def remove_stopwords(self, text):
        """
        Removes selected stopwords from the input text
        Args:
            text (str): Input sentence

        Returns:
            The input sentence without the specified stop-words

        """
        stop_words_list = ['the', 'i', 'to', 'is', 'a', 'it', 'and', 'you', 'in', 'that', 'of', 'this',
                           'have', 'for', 'with', 'on', 'am',
                           'are', 'if', 'my', 'an', 'as', 'would', 'your', 'there', 'has', 'then']
        self.text = text
        for word in stop_words_list:
            pattern = r'\b' + word + r'\b'
            self.text = re.sub(pattern, '', self.text)
        return self.text

    def removal_punction(self, text):
        """
        Removes punctuation
        Args:
            text:

        Returns:
            Text without punctuations

        """
        punction = '"#$%&\'()*+,-/:;<=>[\]^_`{|}~'
        self.text = text
        for p in punction:
            self.text = self.text.replace(p, '')
        return self.text

    def preprocess_text(self, sentence):
        """

        Args:
            sentence (str): Input sentence

        Returns:
            A clean str without URLs, selective punctuations,contractions, selective stopwords.

        """
        self.sentence = sentence.lower()
        # strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        self.sentence = self.decontract(self.sentence)
        self.sentence = self.remove_stopwords(self.sentence)
        self.sentence = self.remove_pattern(self.sentence, "@[\w]*")  # Replace words starting with'@'
        self.sentence = re.sub(r'http\S+', '', self.sentence)
        self.sentence = re.sub(r'www\S+', '', self.sentence)
        self.sentence = re.sub("^[0-9]+$", '', self.sentence)
        # sentence = re.sub(strip_special_chars,'', sentence)
        self.sentence = self.removal_punction(self.sentence)
        self.sentence.strip()
        return self.sentence

    def convert_string(self, w2v_name):
        """
        Assigns the specified Word2Vec model name to the respective embedding model
        Args:
            w2v_name (str): Word2Vec model name

        Returns:
            The fully-loaded Word2Vec model is returned.

        """
        if w2v_name == "w2so_model":
            w2v_name = w2so_model
        else:
            w2v_name = w2gn_model
        return w2v_name

    def tokenizing_embedding(self, embedding_dimensions, sentences, w2v_name):
        """
        Input sentence is tokenized and padded. Words present in the input sentence is
        matched with the vector weight in the Word2Vec model. If there exists a word that is not
        found in the word embedding model, then zero vector of size equal to embedding_dimensions is allotted.

        Args:
            embedding_dimensions (int): Dimensions of the embedding model
            sentences (str): Sentence input by user.
            w2v_name (str): Name of the Word2Vec model

        Returns:
            Padded input sentence.

        """
        print("inside tokenining_embedding")
        w2v_name = self.convert_string(w2v_name)
        df_entire, labels = self.load_df_labels()
        df_entire['Sentence'] = df_entire['Sentence'].apply(lambda x: self.preprocess_text(x))
        training_sentences = df_entire['Sentence'].to_list()

        vocab_size = len(w2v_name.key_to_index)
        embedding_dimensions = embedding_dimensions
        print("printing embedding dimension in tokenizing()")
        print(embedding_dimensions)
        max_length = 135
        trunc_type = 'post'
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(training_sentences)

        # Get our training data word index
        word_index = tokenizer.word_index
        print('found %s unique token' % len(word_index))
        train_sequences = tokenizer.texts_to_sequences(sentences)
        print(train_sequences)
        # print("trainfijng sequencesssssssssss")
        # print(train_sequences)
        train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post")
        print(train_padded)

        # we will map embeddings from the loaded word2vec model
        # for each word to the tokenizer_obj.word_index vocabulary and create a matrix with of word vectors.
        num_word = len(word_index) + 1
        self.embedding_matrix = np.zeros((num_word, embedding_dimensions))
        print(embedding_dimensions)
        print(word_index)
        for word, i in word_index.items():
            try:
                if i > num_word:
                    continue
                self.embedding_vector = w2v_name[word]
                if self.embedding_vector is not None:
                    self.embedding_matrix[i] = self.embedding_vector

                    print(self.embedding_matrix[i].shape)
            except KeyError:
                print("jddkjfdjf")
                print(embedding_dimensions)
                self.embedding_vector = np.zeros((embedding_dimensions,))

        return train_padded

    def tokenizing_embedding_lstm_cnn(self, embedding_dimensions, sentences, w2v_name):
        """

        Args:
            embedding_dimensions:
            sentences:
            w2v_name:

        Returns:

        """
        print("I am in tokenizing_embedding_lstm_cnn--")
        print(w2v_name)
        w2v_name = self.convert_string(w2v_name)
        print(w2v_name)
        self.df_entire, self.labels = self.load_df_labels()
        self.df_entire['Sentence'] = self.df_entire['Sentence'].apply(lambda x: self.preprocess_text(x))
        training_sentences = self.df_entire['Sentence'].to_list()

        vocab_size = len(w2v_name.key_to_index)
        print(vocab_size)
        embedding_dimensions = embedding_dimensions
        print("printing embedding dimension in tokenizing()")
        print(embedding_dimensions)
        max_length = 135
        trunc_type = 'post'
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(training_sentences)

        # Get our training data word index
        word_index = tokenizer.word_index
        print('found %s unique token' % len(word_index))
        train_sequences = tokenizer.texts_to_sequences(sentences)
        print(train_sequences)
        # print("trainfijng sequencesssssssssss")
        # print(train_sequences)
        train_padded = pad_sequences(train_sequences, maxlen=max_length)
        print("printinggg train padded")
        print(train_padded)

        # we will map embeddings from the loaded word2vec model
        # for each word to the tokenizer_obj.word_index vocabulary and create a matrix with of word vectors.
        num_word = len(word_index) + 1
        self.embedding_matrix = np.zeros((num_word, embedding_dimensions))
        print(embedding_dimensions)
        print(word_index)
        for word, i in word_index.items():
            try:
                if i > num_word:
                    continue
                self.embedding_vector = w2v_name[word]
                if self.embedding_vector is not None:
                    self.embedding_matrix[i] = self.embedding_vector

                    print(self.embedding_matrix[i].shape)
            except KeyError:
                # rint ("not found! ",  word)
                print("jddkjfdjf")
                print(embedding_dimensions)
                self.embedding_vector = np.zeros((embedding_dimensions,))
            # print(embedding_vector)
        return train_padded

    def model_load(self, model_name, w2v_name):
        """
        Loads the deep-learning classifier model specified
        Args:
            model_name (str): Name of the deep-learning classifier
            w2v_name (str): Name of the Word2Vec model.

        Returns:
            Loaded Model and the dimensions of the embeddings.

        """
        try:
            if w2v_name == "w2so_model" and model_name == "CNN":
                self.model = load_model("./final_models/model_cnn_100_l" + ".h5", compile=True)
                self.embedding_dim = 100
            elif w2v_name == "w2gn_model" and model_name == "CNN":
                self.model = load_model("./final_models/model_cnn_300_l" + ".h5", compile=True)
                self.embedding_dim = 300
            elif w2v_name == "w2so_model" and model_name == "BILSTM":
                self.model = load_model("./final_models/bilstm_1004" + ".h5", compile=True)
                self.embedding_dim = 100
            elif w2v_name == "w2gn_model" and model_name == "BILSTM":
                self.model = load_model("./final_models/bilstm_3001" + ".h5", compile=True)
                self.embedding_dim = 300
            elif w2v_name == "w2so_model" and model_name == "LSTM":
                self.model = load_model("./final_models/lstm_100" + ".h5", compile=True)
                self.embedding_dim = 100
            elif w2v_name == "w2gn_model" and model_name == "LSTM":
                self.model = load_model("./final_models/lstm_300" + ".h5", compile=True)
                self.embedding_dim = 300
            else:
                #                self.model = load_model("./models/cnn_300"+".h5", compile = True)
                #                self.embedding_dim = 300
                print("No model!")

            print(self.model.summary())
            return self.model, self.embedding_dim
        except:
            print("model not available")

    def sentiment_sentence(self, sentence, model_name, w2v_name):
        """
        Identifies the sentiment of a given input sentence.
        Args:
            sentence (str): Input sentence
            model_name (str): Name of the deep-learning classifier
            w2v_name (str): Name of Word2Vec model

        Returns:
            Sentiment of the input sentence as a str.

        """
        self.model, self.embedding_dim = self.model_load(model_name, w2v_name)
        print("i am in classifyyyy_sentence")
        # print(sent_padded)
        sent_list = [sentence]
        clean_list = list(map(lambda x: self.preprocess_text(x), sent_list))

        if model_name == "LSTM":
            print(model_name)
            self.padded = self.tokenizing_embedding_lstm_cnn(self.embedding_dim, clean_list, w2v_name)
            print("i am in self_padded")
            self.predict_sent = np.argmax(self.model.predict(self.padded), axis=-1)
            print(self.predict_sent)
            if self.predict_sent == 0:
                classify_class = "negative"
                print(classify_class)
            elif self.predict_sent == 1:
                classify_class = "neutral"
                print(classify_class)
            else:
                classify_class = "positive"
                print(classify_class)
        else:
            print(model_name)
            print("elseclause")
            self.padded = self.tokenizing_embedding(self.embedding_dim, clean_list, w2v_name)
            self.predict_sent = np.argmax(self.model.predict(self.padded), axis=-1)
            print(self.predict_sent)
            if self.predict_sent == 0:
                classify_class = "negative"
            elif self.predict_sent == 1:
                classify_class = "neutral"
            else:
                classify_class = "positive"

        return classify_class
