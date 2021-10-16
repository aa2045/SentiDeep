
import pandas as pd
import re
import matplotlib.pyplot as plt
from detect_delimiter import detect

# Load the word embedding models as global variables


class LoadData:
    """Reads the specified csv file into pandas dataframe and cleans the text contained in it

    """

    def __init__(self):
        pass

    # read csv file
    def read_dataset(self, csvfile):
        """Reads the csv file into a pandas dataset

        Args:
            csvfile (str): Name of the csv file

        Returns:
            Dataset containing the samples and labels

        """
        try:
            with open(csvfile) as dataset_file:
                print(type(csvfile))
                print("CSVFILE DATATYPE")
                firstline = dataset_file.readline()
            dataset_file.close()
            # checks the delimeter
            delimeter = detect(firstline)
            print(delimeter)
            df_entire = pd.read_csv(csvfile, sep=delimeter)
            df_entire = df_entire.dropna()
            if df_entire.empty==True:
                print("File is empty")

            else:
                print("valid dataset")
                print(type(df_entire))
                print("datatype of df")
                return df_entire
        except:
            print("No valid file entered")

    # to visualize the class distribution of the dataset
    def visualisation_of_dataset(self, dataset):
        """Plots the sentiment class distribution of a dataset

        Args:
            dataset (LoadData): Dataset containing the labels and samples
        """
        print("Class Distribution")
        x = dataset.groupby('Label').size()
        a = len(dataset['Label'])
        c = (x / a) * 100
        print(c.astype(str) + "%")
        tx = (dataset['Label'].value_counts(normalize=True, sort=True) * 100).plot.bar()
        tx.set(ylabel="Percentage")
        tx.set_title('dataset class distribution')
        tx.legend(labels=['Sentiment Label'])
        plt.show()

    def decontract(self, text):
        """Expands contraction words in text

        Args:
            text (str): Input sentence

        Returns:
            Sentence with expanded contractions

        """
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub("1st", "first", text)
        text = re.sub("2nd", "second", text)
        text = re.sub("3rd", "third", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"isn\'t", "is not", text)
        text = re.sub(r"doesn\'t", "does not", text)
        text = re.sub(r"didn\'t", "did not", text)
        text = re.sub(r"wasn\'t", "was not", text)
        text = re.sub(r"shouldn\'t", "should not", text)
        text = re.sub(r"hasn\'t", "has not", text)
        return text

    def remove_pattern(self, input_txt, pattern):
        """Removes words with the specified pattern, example words starting with '@' in the case of usernames

        Args:
            input_txt (str): Input sentence
            pattern (str): pattern specified to remove

        Returns:
            Sentence without the specified pattern words

        """
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    # method to remove stop-words
    def remove_stopwords(self, text):
        """Removes selective stopwords from the sentence which is specified in a list

        Args:
            text (str): Input Sentence

        Returns:
            Sentence without the defined stopwords

        """
        # conservative stop word list
        stop_words_list = ['the', 'i', 'to', 'is', 'a', 'it', 'and', 'you', 'in', 'that', 'of', 'this',
                           'have', 'for', 'with', 'on', 'am',
                           'are', 'if', 'my', 'an', 'as', 'would', 'your', 'there', 'has', 'then']
        for word in stop_words_list:
            pattern = r'\b' + word + r'\b'
            text = re.sub(pattern, '', text)
        return text

    def removal_punctuation(self, text):
        """ Selective punctuations are removed

        Args:
            text (str): Input sentence

        Returns:
            Text without the selective punctuations

        """

        punctuation = '"#$%&\'()*+,-/:;<=>[\]^_`{|}~'
        for p in punctuation:
            text = text.replace(p, '')
        return text

    def preprocess_text(self, sentence):
        """Preprocessing text

        Args:
            sentence (str): Input sentence

        Returns:
            Preprocessed sentence

        """
        sentence = sentence.lower()
        sentence = self.decontract(sentence)
        sentence = self.remove_stopwords(sentence)
        sentence = self.remove_pattern(sentence, "@[\w]*")  # Replace words starting with'@'
        sentence = re.sub(r'http\S+', '', sentence)
        sentence = re.sub(r'www\S+', '', sentence)
        sentence = re.sub("^[0-9]+$", '', sentence)
        sentence = self.removal_punctuation(sentence)
        sentence.strip()
        return sentence
