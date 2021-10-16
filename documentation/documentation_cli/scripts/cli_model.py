import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from keras.layers import Dense , Input , LSTM , Embedding, Dropout, Bidirectional, Flatten
from keras.initializers import Constant
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, save_model, load_model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tokenizing_labels import Tokenize
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import make_classification
import numpy as np


class ModelTrain:
    """Class consists of methods to create the LSTM, BILSTM and CNN models
    consist of method to perform Stratified k fold cross validation and obtain evaluation results

    """
    def __init__(self, dataset):
        self.tokenize = Tokenize(dataset)

    #BILSTM model
    def model_bilstm(self, num_word, embedding_dim, embedding_matrix, max_length):
        """Builds stacked 2-layer BiLSTM model

        Args:
            num_word (int): length of the vocabulary in Word2Vec model
            embedding_dim (int): dimension of Word2Vec model
            embedding_matrix (int): embedding matrix for sample in dataset
            max_length (int): maximum number of inputs the model can have is the average length of the dataset samples

        Returns:
            Stacked 2-layer BiLSTM model

        """
        model = Sequential()
        embedding_layer = Embedding(num_word, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                                    input_length=max_length, trainable=False)
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(16, dropout=0.3, recurrent_dropout=0.1, kernel_regularizer=regularizers.l2(0.001),
                                      return_sequences=True)))
        model.add(
            Bidirectional(LSTM(16, dropout=0.3, recurrent_dropout=0.1, kernel_regularizer=regularizers.l2(0.001))))

        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        opt = keras.optimizers.RMSprop(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    #LSTM model
    def model_lstm(self, num_word, embedding_dim, embedding_matrix, max_length):
        """Builds stacked 2-layer LSTM model

        Args:
            num_word (int): length of the vocabulary in Word2Vec model
            embedding_dim (int): dimension of Word2Vec model
            embedding_matrix (int): embedding matrix for sample in dataset
            max_length (int): maximum number of inputs the model can have is the average length of the dataset samples

        Returns:
            Stacked 2-layer LSTM model

        """
        model = Sequential()
        embedding_layer = Embedding(num_word, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                                    input_length=max_length, trainable=False)
        model.add(embedding_layer)
        model.add(LSTM(16, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.001),
                       return_sequences=True))
        model.add(LSTM(16, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(8, activation='relu'))

        model.add(Dense(3, activation='softmax'))
        opt = keras.optimizers.RMSprop(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    #CNN Model
    def model_cnn(self, num_word, embedding_dim, embedding_matrix, max_length):
        """Builds stacked 2-layer CNN model

        Args:
            num_word (int): length of the vocabulary in Word2Vec model
            embedding_dim (int): dimension of Word2Vec model
            embedding_matrix (int): embedding matrix for sample in dataset
            max_length (int): maximum number of inputs the model can have is the average length of the dataset samples

        Returns:
            Stacked 2-layer CNN model

        """
        model = Sequential()
        embedding_layer = Embedding(num_word, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                                    input_length=max_length, trainable=False)
        model.add(embedding_layer)
        model.add(Conv1D(16, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
        model.add(Conv1D(8, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
        model.add(MaxPooling1D())
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
        opt = opt = tf.keras.optimizers.SGD(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(model.summary())
        return model

    #method when called will create the specified model
    def create_model(self, model_name, num_word, embedding_dim, embedding_matrix, max_length):
        """Calling function, to build the specified deep-learning classifier

        Args:
            model_name (str): Name of the classifier model
            num_word (int): length of the vocabulary in Word2Vec model
            embedding_dim (int): dimension of Word2Vec model
            embedding_matrix (int): embedding matrix for sample in dataset
            max_length (int): maximum number of inputs the model can have is the average length of the dataset samples

        Returns:
            The specified Deep-learning model

        """

        if (model_name == "BILSTM"):
            model = self.model_bilstm(num_word, embedding_dim, embedding_matrix, max_length)
            return model
        elif (model_name == "LSTM"):
            model = self.model_lstm(num_word, embedding_dim, embedding_matrix, max_length)
            return model
        elif (model_name == "CNN"):
            model = self.model_cnn(num_word, embedding_dim, embedding_matrix, max_length)
            return model
        else:
            print("error in model laoding")
            return

    #saves the models developed in the specified directory
    def directory_name(self, model_name, embedding_dimensions):
        """Creates a directory to save the trained models from each fold

        Args:
            model_name (str): Name of the classifier model
            embedding_dimensions (int): Embedding dimension of the word embedding model

        Returns:
            Name of the directory

        """
        directory = './' + model_name + str(embedding_dimensions) + '/'
        print(directory)
        return directory

    def get_model_name(self, model_name):
        """Creates a name to save the trained model

        Args:
            model_name (str): Name of the classifier model

        Returns:
            Name of the saved model

        """
        name_model = model_name
        return name_model


    def call_tokenize(self, model_name, embedding_dimensions, embed_model):
        """Calling function, that calls the tokenizer functions of Tokenize class to tokenize, pad and
        create embedding matrices for the input sample.

        Args:
            model_name (str): Name of the classifier model
            embedding_dimensions (int): Embedding dimension of the word embedding model
            embed_model (str): Name of the Word2Vec model

        Returns:
            Embedding matrix, padded samples, encoded labels, length of the vocabulary and the mean length of the samples

        """
        if (model_name == "BILSTM" or "CNN"):
            embedding_matrix, train_padded, labels, num_word, max_length = self.tokenize.tokenizing_embedding_cnn_bilstm(
                embedding_dimensions, embed_model)
            return embedding_matrix, train_padded, labels, num_word, max_length
        elif (model_name == "LSTM"):
            embedding_matrix, train_padded, labels, num_word, max_length = self.tokenize.tokenizing_embedding_lstm(
                embedding_dimensions, embed_model)
            return embedding_matrix, train_padded, labels, num_word, max_length
        else:
            print("error")
            return


    def plot_confusion_matrix(self, cn_matrix):
        """Plots the confusion matrix

        Args:
            cn_matrix (int): Confusion Matrix of the model


        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(cn_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(cn_matrix.shape[0]):
            for j in range(cn_matrix.shape[1]):
                ax.text(x=j, y=i, s=cn_matrix[i, j], va='center', ha='center', size='xx-large')
        return

    def validation_plot(self,history, metric):
        """Plots the specified loss/accuracy  plot of a model

        Args:
            history (obj): holds loss values and metric values during training
            metric (str): accuracy or loss values of the model
        """
        plt.plot(history.history[metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.show()

    #Stratified k fold cross validation method
    def k_fold_train(self, model_name, embed_model, epochs_num):
        """

        Args:
            model_name (str): Name of the deep-learning classifier model
            embed_model (str): Name of the Word2Vec Model
            epochs_num (int): Number of epochs

        Returns:
            List containing the values of evaluation metrics (Precision, Recall, F-measure) through all folds

        """
        if embed_model == "SO":
            embedding_dimensions = 100
        elif embed_model == "Google News":
            embedding_dimensions = 300
        else:
            print("k_fold_train, embed model wrong")

        print(embedding_dimensions)
        save_dir = self.directory_name(model_name, embedding_dimensions)
        embedding_matrix, X, y, num_word, max_length = self.call_tokenize(model_name, embedding_dimensions, embed_model)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        fold_no = 0
        report_list = []
        for train_index, test_index in skf.split(X, y):
            # train and test data
            X_train, X_test = X[train_index], X[test_index]
            # converting label encoded values to one-hot encoding values
            y_categorical = to_categorical(y)

            # train and test labels
            y_train, y_test = y[train_index], y[test_index]
            y_train = to_categorical(y_train)

            #model is created
            model = self.create_model(model_name, num_word, embedding_dimensions, embedding_matrix, max_length)
            # checkpoint to save the model
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                save_dir + self.get_model_name(model_name) + str(fold_no) + '.h5', verbose=1)

            # train the model
            history = model.fit(X_train, y_train, batch_size=64,
                                shuffle=True, epochs= int(epochs_num),
                                callbacks=[checkpoint])
            print(str(fold_no))
            #load the saved model
            model1 = load_model(save_dir + self.get_model_name(model_name) + str(fold_no) + '.h5', compile=True)
            #validate on the test set
            predictions = np.argmax(model1.predict(X_test), axis=-1)
            #confusion matrix
            cn_matrix = confusion_matrix(y_test, predictions)
            print(cn_matrix)

            print("CONFUSION MATRIX")
            print(cn_matrix)

            print('\n clasification report:\n', classification_report(y_test, predictions))
            report = classification_report(y_test, predictions, output_dict=True)
            report_list.append(report)

            #plotting training accuracy and training loss
            print(self.validation_plot(history, 'accuracy'))
            print(self.validation_plot(history, 'loss'))


            #clears the session
            tf.keras.backend.clear_session()
            #increments the fold
            fold_no = fold_no + 1

        #placeholders to store precision, recall, f-score values for 5 folds
        precision_0 = []
        precision_1 = []
        precision_2 = []

        recall_0 = []
        recall_1 = []
        recall_2 = []

        fscore_0 = []
        fscore_1 = []
        fscore_2 = []

        for i in report_list:
            precision_0.append(i['0']['precision'])
            precision_1.append(i['1']['precision'])
            precision_2.append(i['2']['precision'])

            recall_0.append(i['0']['recall'])
            recall_1.append(i['1']['recall'])
            recall_2.append(i['2']['recall'])

            fscore_0.append(i['0']['f1-score'])
            fscore_1.append(i['1']['f1-score'])
            fscore_2.append(i['2']['f1-score'])
        print(
            "--------------------------------------------------------------------------------------------------------------------------")
        print("                                 Report                            ")

        print(
            "--------------------------------------------------------------------------------------------------------------------------")
        print(
            "Fold         Overall            Negative Neutral Positive       Negative Neutral Positive         Negative Neutral Positive ")
        print(
            "--------------------------------------------------------------------------------------------------------------------------")
        print(
            "Fold         P | R | F                   Precision                       Recall                                F1 ")
        print(
            "--------------------------------------------------------------------------------------------------------------------------")

        for i in range(0, fold_no):
            p = (precision_0[i] + precision_1[i] + precision_2[i]) / 3
            r = (recall_0[i] + recall_1[i] + recall_2[i]) / 3
            f = (fscore_0[i] + fscore_1[i] + fscore_2[i]) / 3
            print(i, " ", "", '%0.6f' % p, "|", '%0.6f' % r, "|", '%0.6f' % f, "  ", '%0.6f' % precision_0[i],
                  " "'%0.6f' % precision_1[i], " "'%0.6f' % precision_2[i], "  "'%0.6f' % recall_0[i],
                  '%0.6f' % recall_1[i], '%0.6f' % recall_2[i], " "'%0.6f' % fscore_0[i], '%0.6f' % fscore_1[i],
                  '%0.6f' % fscore_2[i])
        print(
            "---------------------------------------------------------------------------------------------------------------------------")

        print("Avg. ", '%0.6f' % np.mean(p), '%0.6f' % np.mean(r), '%0.6f' % np.mean(f), "    ",
              '%0.6f' % np.mean(precision_0), " "'%0.6f' % np.mean(precision_1), " "'%0.6f' % np.mean(precision_2),
              "  "'%0.6f' % np.mean(recall_0), '%0.6f' % np.mean(recall_1), '%0.6f' % np.mean(recall_2),
              "  "'%0.6f' % np.mean(fscore_0), '%0.6f' % np.mean(fscore_1), '%0.6f' % np.mean(fscore_2))
        print("Std. ", "                               ", '%0.6f' % np.std(precision_0),
              " "'%0.6f' % np.std(precision_1), " "'%0.6f' % np.std(precision_2), "  "'%0.6f' % np.std(recall_0),
              '%0.6f' % np.std(recall_1), '%0.6f' % np.std(recall_2), "  "'%0.6f' % np.std(fscore_0),
              '%0.6f' % np.std(fscore_1), '%0.6f' % np.std(fscore_2))
        return report_list