from create_corpus import CombineDataset
import gensim
from gensim.models import KeyedVectors
import numpy as np
import os


class EmbedModel():
    """Class is responsible for creating the Word2Vec model
       from the preprocessed stackoverflow corpus

       Attributes:
           combine_dataset (obj): Object of CombineDataset class
           stack_df (): preprocessed stackoverflow dataset containing questions and answers combined

      Return:
          w2vso_model: Software specific Word2Vec model


    """
    def __init__(self):

        self.combine_dataset = CombineDataset()
        self.stack_df = self.combine_dataset.combine_clean_xml()
        self.create_word2vec_model()

    def create_word2vec_model(self):
        """Creates the word embedding model from the StackOverflow corpus

        Returns:
            Word vectors as output

        """

        # word2vec parameters
        w2v_size = 100
        w2v_window = 5
        w2v_epoch = 3
        w2v_min_count_words = 5

        # Collect the corpus for training word embeddings
        # used stackOverflow posts that contains questions and answers
        # posts are tokenized and appended to a list
        post_corpus = [post_text.split() for post_text in np.array(self.stack_df.post_corpus)]
        # initializing the model
        w2v_model2mill = gensim.models.word2vec.Word2Vec(vector_size=w2v_size,
                                                         window=w2v_window,
                                                         min_count=w2v_min_count_words,
                                                         workers=5)
        # builing a vocabulary from the specified corpus
        w2v_model2mill.build_vocab(post_corpus)
        # training the word2vec model
        w2v_model2mill.train(post_corpus, total_examples=len(post_corpus), epochs=w2v_epoch)
        # saving the word2vec model
        w2v_model2mill.wv.save_word2vec_format('w2v_2m.bin', binary=True)
        cwd_so = os.getcwd()
        print(cwd_so)
        path_so = cwd_so + "\w2v_2m.bin"
        print(path_so)
        # loading the developed word2vec model
        w2so_model = gensim.models.KeyedVectors.load_word2vec_format(path_so, binary=True, unicode_errors='ignore')
        print(w2so_model.most_similar("console"))
        print(len(w2so_model.key_to_index))
        return w2so_model


if __name__ == "__main__":
    f = EmbedModel()