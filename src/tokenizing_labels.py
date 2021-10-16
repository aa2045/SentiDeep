import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import LoadData
import gensim
from gensim.models import KeyedVectors
import os

# Load the word embedding models as global variables
# Google news embedding
cwd_gn = os.getcwd()
print(cwd_gn)
path_gn = cwd_gn+"\GoogleNews-vectors-negative300.bin.gz"
print(path_gn)
w2gn_model = gensim.models.KeyedVectors.load_word2vec_format(path_gn, binary=True,unicode_errors='ignore')

#StackOverflow embedding model
cwd_so = os.getcwd()
print(cwd_so)
path_so = cwd_so+"\w2v_100_SO.bin"
print(path_so)
w2so_model = gensim.models.KeyedVectors.load_word2vec_format(path_so, binary=True,unicode_errors='ignore')


class Tokenize:
    """
    Class is responsible for tokenzing the samples in dataset and encoding the labels

    """

    def __init__(self, dataset):

        # loads the dataset
        self.l = LoadData()
        self.df_entire = self.l.read_dataset(dataset)
        print(self.l.visualisation_of_dataset(self.df_entire))
        # self.p = Preprocess()
        # preprocess the dataset
        self.df_entire['Sentence'] = self.df_entire['Sentence'].apply(lambda x: self.l.preprocess_text(x))

    # calculate the average sentence length
    def calc_average(self):
        """Calculates the average length of all sentences present

        Returns:
            Average length

        """
        self.df_entire['Sentence_length'] = self.df_entire['Sentence'].str.len()
        mean = self.df_entire['Sentence_length'].mean()
        avg_length = int(round(mean, 0))
        return avg_length

    # returns encoded labels and list of sentences
    def label_encoding(self):
        """Encodes the labels in the dataset

        Returns:
            Encoded labels

        """
        training_sentences = self.df_entire['Sentence'].to_list()
        training_labels = self.df_entire['Label'].to_list()
        # encoding labels for train and test set
        label_encoder = LabelEncoder()
        encoded_label = np.array(label_encoder.fit_transform(training_labels))
        # inverse labels
        inverse_Y = label_encoder.inverse_transform(encoded_label)
        print(inverse_Y)
        label_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(label_name_mapping)
        return training_sentences, encoded_label

    # this method tokenizes, and pads the text.
    # embeddings are mapped to the word2vec model
    # returns padded text, labels, embedding matrix, max_length and number of words
    def tokenizing_embedding_cnn_bilstm(self, embedding_dimensions, embed_model):
        """
        Input sentence is tokenized and padded. Words present in the input sentence is
        matched with the vector weight in the Word2Vec model. If there exists a word that is not
        found in the word embedding model, then zero vector of size equal to embedding_dimensions is allotted.

        Args:
            embedding_dimensions (int): Dimensions of the embedding model
            embed_model (str): Name of the Word2Vec model

        Returns:
            Padded input sentence, embedding_matrix, labels, number of words in embedding model and
             avg length of sentence

        """
        if embed_model == "SO":
            w2v_model = w2so_model
            embedding_dimensions = 100
        elif embed_model == "Google News":
            w2v_model = w2gn_model
            embedding_dimensions = 300
        else:
            print("embed model something wrong")
        vocab_size = len(w2v_model.key_to_index)
        embedding_dimensions = embedding_dimensions
        max_length = self.calc_average()
        # max number of words to keep is set to the total number of words in word2vec model
        tokenizer = Tokenizer(num_words=vocab_size)
        training_sentences, labels = self.label_encoding()
        # updates the internal vocabulary based on the training sentences which is a list
        tokenizer.fit_on_texts(training_sentences)

        # Get our training data word index
        word_index = tokenizer.word_index
        # each sample is in the list of samples is transorfmed to a sequence of integers
        train_sequences = tokenizer.texts_to_sequences(training_sentences)
        # padding the sequences to the same length
        train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post")
        # sentiment = df_entire['Label'].values

        # we will map embeddings from the loaded word2vec model
        # for each word to the tokenizer_obj.word_index vocabulary and create a matrix with of word vectors.
        num_word = len(word_index) + 1
        embedding_matrix = np.zeros((num_word, embedding_dimensions))
        print(word_index)
        for word, i in word_index.items():
            try:
                if i > num_word:
                    continue
                embedding_vector = w2v_model[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            except KeyError:
                embedding_vector = np.zeros((embedding_dimensions,))

        return embedding_matrix, train_padded, labels, num_word, max_length

    # this method tokenizes, and pads the text.
    # embeddings are mapped to the word2vec model
    # returns padded text, labels, embedding matrix, max_length and number of words
    def tokenizing_embedding_lstm(self, embedding_dimensions, w2v_model):

        vocab_size = len(w2v_model.key_to_index)
        embedding_dimensions = embedding_dimensions
        max_length = self.calc_average()
        # max number of words to keep is set to the total number of words in word2vec model
        tokenizer = Tokenizer(num_words=vocab_size)
        training_sentences, labels = self.label_encoding()
        # updates the internal vocabulary based on the training sentences which is a list
        tokenizer.fit_on_texts(training_sentences)

        # Get our training data word index
        word_index = tokenizer.word_index
        # each sample is in the list of samples is transorfmed to a sequence of integers
        train_sequences = tokenizer.texts_to_sequences(training_sentences)
        # padding the sequences to the same length
        train_padded = pad_sequences(train_sequences, maxlen=max_length)
        # sentiment = df_entire['Label'].values

        # we will map embeddings from the loaded word2vec model
        # for each word to the tokenizer_obj.word_index vocabulary
        # and create a matrix with of word vectors.

        num_word = len(word_index) + 1
        embedding_matrix = np.zeros((num_word, embedding_dimensions))
        print(word_index)
        for word, i in word_index.items():
            try:
                if i > num_word:
                    continue
                embedding_vector = w2v_model[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            except KeyError:
                embedding_vector = np.zeros((embedding_dimensions,))

        return embedding_matrix, train_padded, labels, num_word, max_length
