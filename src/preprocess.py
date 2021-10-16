import pandas as pd
import re


class LoadDataset():

    """ Reads the csv file and creates a pandas dataset.
        Stackoverflow questions have multiple answers for a single question,
        so all answers for each question is grouped.

        :arg
            df (): dataset containing the stackoverflow posts

    """

    def read_dataset(self):
        """Reads the csv file into a Pandas DataFrame

        Returns:
            DataFrame

        """
        try:
            df = pd.read_csv("2milldump.csv")
            print("Stackdatadump is found")
            return df
        except:

            print("Datadump not found, please place it in src directory and run this")

    def group_answers(self):
        """Groups multiple answers for single stackoverflow question
        Returns:
            DataFrame with all answers combined for a stackoverflow question
        """
        self.df = self.read_dataset()
        # stackoverflow posts have multiple answers for a question, and so multiple answers for a question are grouped
        grouped_answers = self.df.groupby(['id', 'title', 'body', 'tags']).agg({'answers': lambda x: "\n".join(x)})
        grouped_answers.columns = ['combined_answers']
        grouped_answers = grouped_answers.reset_index()
        # dataframe that contains all grouped answers for the questions
        grouped_df = pd.DataFrame(grouped_answers)
        print(grouped_df.head(20))
        print(grouped_df.isna().sum())
        return grouped_df


class Preprocess():
    """ Preprocesses the dataset

    Return:
        Tokenized sentences are returned

    """

    def remove_stopwords(self, sentence):

        """Removes the stopwords from the dataset specified in set

        Args:
            sentence (str): Text present in the dataset

        Returns:
            List without the selective stopwords

        """
        stopwords_set = set(['the', 'i', 'to', 'is', 'a', 'it', 'and', 'you', 'in', 'that', 'of', 'this',
                             'have', 'for', 'with', 'on', 'am',
                             'are', 'if', 'my', 'an', 'as', 'would', 'your', 'there', 'has', 'then'])

        clean_stopword_list = [word for word in sentence if word not in stopwords_set]
        return clean_stopword_list

    # removes punctuations and urls
    def remove_punctuation_url(self, sentence):
        """Removes punctuations, URLs, HTTP links and appends the clean text to a list.
           The list is returned.
        Args:
            sentence (str): Sample from the dataframe

        Returns:
            List of samples without the selective punctuations and URLs

        """
        remove_punct_list = []
        for word in sentence:
            # removing punctuations
            new_word = re.sub(r"[^-a-zA-Z.?!]", '', word)
            # removing urls
            new_word = re.sub(r'http\S+', '', new_word)
            new_word = re.sub(r'www\S+', '', new_word)
            remove_punct_list.append(new_word)
        return remove_punct_list

    def lowercase_text(self, sentence):
        """
        Converts the text to lowercase
        Args:
            sentence (str): Input sentence in the sample

        Returns:
            List of all samples after lowercasing it

        """
        # converting text to lowercase
        lowercase_list = [word.lower() for word in sentence]
        return lowercase_list

    def decontract(self, text):

        ''' Contracted words that need to be expanded'''

        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"wont", "will not", text)
        text = re.sub(r"can\'t", "cannot", text)
        text = re.sub(r"cant", "cannot", text)
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
        text = re.sub("it's", "it is", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"isn\'t", "is not", text)
        text = re.sub(r"doesn\'t", "does not", text)
        text = re.sub(r"doesnt", "does not", text)
        text = re.sub(r"didn\'t", "did not", text)
        text = re.sub(r"didnt", "did not", text)
        text = re.sub(r"wasn\'t", "was not", text)
        text = re.sub(r"wasnt", "was not", text)
        text = re.sub(r"shouldn\'t", "should not", text)
        text = re.sub(r"shouldnt", "should not", text)
        text = re.sub(r"hasn\'t", "has not", text)
        text = re.sub(r"hasnt", "has not", text)
        return text

    def decontraction_words(self, sentence):
        """Expands decontracted words

        Args:
            sentence (str): Sample in the dataframe

        Returns:
            list of all samples after expanding the contraction words

        """

        decontract_words = []
        for word in sentence:
            decontract_words.extend(self.decontract(word).split())
        return decontract_words

    def combine_preprocess(self, sentence):
        """Calling function that calls the methods to expand contractions,
        remove punctuations, stopwords  and normalize text
        Args:
            sentence List[str]: Samples present in the DataFrame

        Returns:
            The samples after preprocessing them

        """
        sentence = self.lowercase_text(sentence)
        sentence = self.decontraction_words(sentence)
        sentence = self.remove_punctuation_url(sentence)
        sentence = self.remove_stopwords(sentence)
        return sentence

    def preprocess_text(self, sentence):
        """Caller function that calls the method to preprocess the text and tokenizes them.

        Args:
            sentence List[str]: Samples present in the dataframe

        Returns:
            preprocessed and tokenized samples of the dataset

        """
        return ' '.join(self.combine_preprocess(sentence.split()))
