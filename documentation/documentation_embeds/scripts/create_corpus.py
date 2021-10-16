from preprocess import LoadDataset
from preprocess import Preprocess
from bs4 import BeautifulSoup
import nltk
import pandas as pd

class CombineDataset():

    """Class is responsible for combining the question, answers into one cell.

        Attributes:
            load_dataset (object): Object of LoadDataset class
            df (): DataFrame containing StackOverflow posts
            grouped_df (): DataFrame containing the answers grouped
            preprocess (object): Object of the Preprocess class

        Return:
            DataFrame containing each of the question and answers confined to a cell

    """


    def __init__(self):
         self.load_dataset  = LoadDataset()
         self.grouped_df = self.load_dataset.group_answers()
         self.preprocess = Preprocess()
         #self.stack_df = self.combine_clean()





    def combine_clean_xml(self):
        """
        Method combines the question, answers to one column
        and removes the code tags present using beautiful soup.
            Return:
                DataFrame that contains the questions, answers combined and cleaned forming the StackOverflow corpus

        """

        #placeholders to store questions, answers and tags in a list
        question_list = []
        question_content_list = []
        answers_list = []
        tag_list = []
        complete_corpus_list = []
        count = 0
        counter = 0
        #iterating through the rows of dataframe
        for i, row in self.grouped_df.iterrows():
            question_list.append(row.title)  # append question title
            tag_list.append(row.tags)  # append question tags
            # Processing Questions
            body_content = row.body
            print(body_content)
            #remove the xml tags present in the dataset
            soup = BeautifulSoup(body_content, 'lxml')
            if soup.code: soup.code.decompose()  # Remove the code part present in questions
            #remove the paragraph and pre tags but keeps the text contained with the tags
            paragraph_tag = soup.p
            pre_tag = soup.pre
            clear_text = ''
            if paragraph_tag: clear_text = clear_text + paragraph_tag.get_text()
            if pre_tag: clear_text = clear_text + pre_tag.get_text()
            #question and title is append to the list
            question_content_list.append(str(row.title) + ' ' + str(clear_text))
            count += 1
            print(count)
            # Processing Answers
            answer_content = row.combined_answers
            soup = BeautifulSoup(answer_content, 'lxml')
            if soup.code: soup.code.decompose()  # Remove the code part present in answers
            # remove the paragraph and pre tags but keeps the text contained with the tags
            paragraph_tag = soup.p
            pre_tag = soup.pre
            clear_text = ''
            if paragraph_tag: clear_text = clear_text + paragraph_tag.get_text()
            if pre_tag: clear_text = clear_text + pre_tag.get_text()
            answers_list.append(clear_text)
            complete_corpus_list.append(question_content_list[-1] + ' ' + answers_list[-1])
            counter += 1
            print(counter)
        #creating a dataframe that contains the questions, answers combined
        stack_df = pd.DataFrame(
            {'original_title': question_list, 'post_corpus': complete_corpus_list, 'question_content': question_content_list,
             'tags': tag_list, 'answers_content': answers_list})
        #print(stack_df.head(10))
        #preprocessing the question, answers and post_corpus
        stack_df.question_content = stack_df.question_content.apply(lambda x: self.preprocess.preprocess_text(x))
        stack_df.post_corpus = stack_df.post_corpus.apply(lambda x: self.preprocess.preprocess_text(x))
        stack_df.answers_content = stack_df.answers_content.apply(lambda x: self.preprocess.preprocess_text(x))
        pd.options.display.max_colwidth = 3000
        print(stack_df.head(10))
        print(stack_df['answers_content'].head(10))
        #writes the preprocessed dataset to a file
        stack_df.to_csv('preprocessed2milldump.csv', index=False)
        return stack_df



