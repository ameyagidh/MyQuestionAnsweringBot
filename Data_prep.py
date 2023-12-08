import pandas as pd


class PrepareData:
    def __init__(self, train_tsv: str, dev_tsv: str, test_tsv: str, pos_ans_tsv: str):
        """
        Creates the Necessary datframes required to create the final_data,
        Initialize the object of this class with following args and then call 
        preprocess function to prepare your data.

        ARGS:
            train_tsv: Path to train_tsv file.
            dev_tsv: Path to dev_tsv file.
            test_tsv: Path to test_tsv file.
        """
        self.train_tsv_path = train_tsv
        self.dev_tsv_path = dev_tsv
        self.test_tsv_path = test_tsv
        self.pos_ans_tsv = pos_ans_tsv

        # Initialize variables to store dataframes.
        self.pos_ans_tsv_df = None
        self.train_tsv_df = None
        self.dev_tsv_df = None
        self.test_tsv_df = None

        # Initialize variables to store final train, dev, test dataframes.
        self.final_train_df = None
        self.final_dev_df = None
        self.final_test_df = None

    def create_data_frames(self):
        # Creates the train, dev, test, pos_ans dataframes.
        self.pos_ans_tsv_df = pd.read_csv(self.pos_ans_tsv, delimiter="\t")
        self.train_tsv_df = pd.read_csv(self.train_tsv_path, delimiter="\t")
        self.dev_tsv_df = pd.read_csv(self.dev_tsv_path, delimiter="\t")
        self.test_tsv_df = pd.read_csv(self.test_tsv_path, delimiter="\t")

    def create_final_df(self, df: pd.DataFrame):
        # Takes in a dataframe and creates a new df ready for training
        final_dict = {
            'question': [],
            'sentence': [],
            'answer': []
        }

        for i, row in df.iterrows():
            # Iterate through each row and create new row for each of correct answer.
            if pd.notna(row['AnswerPhrase1']) and row['AnswerPhrase1'] != 'NO_ANS':
                final_dict['question'].append(row['Question'])
                final_dict['sentence'].append(row['Sentence'])
                final_dict['answer'].append(row['AnswerPhrase1'])

            if pd.notna(row['AnswerPhrase2']) and row['AnswerPhrase2'] != 'NO_ANS':
                final_dict['question'].append(row['Question'])
                final_dict['sentence'].append(row['Sentence'])
                final_dict['answer'].append(row['AnswerPhrase2'])

            if pd.notna(row['AnswerPhrase3']) and row['AnswerPhrase3'] != 'NO_ANS':
                final_dict['question'].append(row['Question'])
                final_dict['sentence'].append(row['Sentence'])
                final_dict['answer'].append(row['AnswerPhrase3'])

        return pd.DataFrame(final_dict)

    def Preprocess(self):
        # Creates the final cleaned datasets and stored them in disk.
        self.create_data_frames()  # Create dataframes.
        # Preprocess the dataframes(eliminate all rows with incorrect sentences.)
        cleaned_train_tsv_data = self.train_tsv_df[self.train_tsv_df["Label"] == 1]
        cleaned_dev_tsv_data = self.dev_tsv_df[self.dev_tsv_df["Label"] == 1]
        cleaned_test_tsv_data = self.test_tsv_df[self.test_tsv_df["Label"] == 1]

        # Get list of all unique_questions in the cleaned data.
        unique_questions_train = cleaned_train_tsv_data['QuestionID'].unique()
        unique_questions_dev = cleaned_dev_tsv_data['QuestionID'].unique()
        unique_questions_test = cleaned_test_tsv_data['QuestionID'].unique()

        # separate train_pos_ans_df for each of train, dev, test.
        train_pos_ans_tsv = self.pos_ans_tsv_df[self.pos_ans_tsv_df['QuestionID'].isin(
            unique_questions_train)]
        dev_pos_ans_tsv = self.pos_ans_tsv_df[self.pos_ans_tsv_df['QuestionID'].isin(
            unique_questions_dev)]
        test_pos_ans_tsv = self.pos_ans_tsv_df[self.pos_ans_tsv_df['QuestionID'].isin(
            unique_questions_test)]

        # Merge train, dev, test dataframes with pos_ans dataframes.
        merged_train_df = pd.merge(train_pos_ans_tsv, cleaned_train_tsv_data, on=[
                                   "QuestionID", "Question", "DocumentID", "DocumentTitle", "SentenceID", "Sentence"])
        merged_dev_df = pd.merge(dev_pos_ans_tsv, cleaned_dev_tsv_data, on=[
                                 "QuestionID", "Question", "DocumentID", "DocumentTitle", "SentenceID", "Sentence"])
        merged_test_df = pd.merge(test_pos_ans_tsv, cleaned_test_tsv_data, on=[
                                  "QuestionID", "Question", "DocumentID", "DocumentTitle", "SentenceID", "Sentence"])

        # Create the final dataframes.
        self.final_train_df = self.create_final_df(merged_train_df)
        self.final_dev_df = self.create_final_df(merged_dev_df)
        self.final_test_df = self.create_final_df(merged_test_df)

        # Save the final dataframes in a csv file.
        self.final_train_df.to_csv("data/train.csv", index=False)
        self.final_dev_df.to_csv("data/dev.csv", index=False)
        self.final_test_df.to_csv("data/test.csv", index=False)
