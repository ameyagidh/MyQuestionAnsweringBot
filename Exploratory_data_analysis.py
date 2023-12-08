"""
Ameya Santosh Gidh.

This file contains code to perform exporatory data analysis.
"""

import numpy as np
import pandas as pd


class ExploratoryAnalysis:
    def __init__(self, train_tsv: str, dev_tsv: str, test_tsv: str):
        """
        Initializes the train, dev, test paths with class variables.

        ARGS:
            train_tsv: Path to train_tsv file.
            dev_tsv: Path to dev_tsv file.
            test_tsv: Path to test_tsv file.
        """
        self.train_tsv_path = train_tsv
        self.dev_tsv_path = dev_tsv
        self.test_tsv_path = test_tsv

    def exploreData(self, data_path):
        # Read and display the file as pandas dataFrame.
        df = pd.read_csv(data_path, delimiter='\t')
        print(df.head(5))

        # Get the number of data in the data.
        print(f"The number of data points in train_tsv file are {df.shape[0]}")

        # Check for the number of questions.
        unique_questions = df['QuestionID'].unique()
        print(f"The number of unique questions are  {len(unique_questions)}")

        # check the count of no.of documents.
        no_of_documents = df['DocumentID'].unique()
        print(
            f"The number of documents in train_tsv file is {len(no_of_documents)}")

        # find the average no.of questions per document.
        avg_questions_per_doc = df.groupby(
            'DocumentID')['QuestionID'].nunique().mean()
        # Print the result
        print("Average number of unique QuestionID values per DocumentID:",
              avg_questions_per_doc)

        # get list of documents with more than one question.
        docs_with_multiple_questions = df.groupby('DocumentID').filter(
            lambda x: x['QuestionID'].nunique() > 1)['DocumentID'].unique()
        print(docs_with_multiple_questions)
        print(
            f"No of documents with more than one question is {len(docs_with_multiple_questions)} which is {len(no_of_documents) / len(docs_with_multiple_questions)}%")

        # Find the average number of answers per question.
        average_no_of_answers_per_question = df.groupby(
            'QuestionID')['SentenceID'].nunique().mean()
        print(
            f"Average no.of answers per question {average_no_of_answers_per_question}")

    def exploreTrainTsv(self):
        print(f"\nExploratory DataAnalysis of train_tsv data")
        self.exploreData(self.train_tsv_path)

    def exploreDevTsv(self):
        print(f"\nExploratory DataAnalysis of dev_tsv data")
        self.exploreData(self.dev_tsv_path)

    def exploreTestTsv(self):
        print(f"\nExploratory DataAnalysis of test_tsv data")
        self.exploreData(self.test_tsv_path)
