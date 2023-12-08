import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
import pandas as pd
import os
from tqdm import tqdm
from utils import valid_string


class Predictions:
    def __init__(self, model: tf.Module, tokenizer: BertTokenizer):
        """
        Initializes the class with model and tokenizer values.
        model: The Pretrained model.
        tokenizer: tokenizer with fine-tuned model weights.
        """
        self.model = model
        self.tokenizer = tokenizer

    def preprocess_input(self, question: str, context: str, tokenizer: BertTokenizer, sequence_length: int):
        """
        Preprocesses the data and converts them into formats of input_ids, attention_masks, token_type_ids.

        ARGS:
            question: question which needs to be processed.
            context: context that needs to be processed.
            tokenizer: pretrained tokenizer with weights of pretrained model.
            sequence_length: sequence_length of each word.

        Returns:
            returns a tuple of input_ids, attention_mask, token_type_ids.
        """

        encoded_data = tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=sequence_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="tf",
        )

        input_ids = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]
        token_type_ids = encoded_data["token_type_ids"]
        return input_ids, attention_mask, token_type_ids

    def get_answer(self, context: str, start_pos: int, end_pos: int, tokenizer: BertTokenizer):
        """
        Extracts the final predicted answer based on start and end idx.

        ARGS:
            context: The context in which answer is present.
            start_pos: Start index of the answer in context.
            end_pos: end index of the answer in context.

        Returns:
            returns the final predicted answer.
        """
        context_tokens = tokenizer.tokenize(context)
        answer_tokens = context_tokens[start_pos:end_pos+1]
        answer_text = tokenizer.convert_tokens_to_string(answer_tokens)

        return answer_text

    def make_prediction(self, question: str, context: str, sequence_length: int = 384):
        """
        Displays the final answer by taking input question and context as parameter.

        ARGS:
            question: Input question.
            context: Input context.
            sequence_length: Max sequence to be taken for each word.

        Returns:
            returns the final answer if found.
        """
        input_ids, attention_mask, token_type_ids = self.preprocess_input(
            question, context, self.tokenizer, sequence_length)
        start_logits, end_logits = self.model.predict(
            [input_ids, attention_mask])

        start_position = np.argmax(start_logits)
        end_position = np.argmax(end_logits)

        # Check if the end position is greater than or equal to the start position, otherwise return an empty string
        if end_position >= start_position:
            answer = self.get_answer(context, start_position,
                                     end_position, self.tokenizer)
        else:
            answer = ""

        return answer


class Evaluations:
    def __init__(self, train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, eval_df: pd.DataFrame, model: tf.Module, tokens: BertTokenizer, pred_object: Predictions):
        """
        Initializes the class with train, dev, test, fine-tuned
        model, fine-tuned tokenizers and creates the final data required for 
        performing evaluations.

        ARGS:
            train_df: cleaned train data_frame.
            dev_df: cleaned dev data_frame.
            test_df: cleaned test data_frame.
            model: fine-tuned model.
            tokens: fine-tuned tokenizer.
            pred_object: Predictions class object.
        """
        self.train_df = train_df
        self.dev_df = dev_df
        self.test_df = test_df
        self.eval_df = eval_df
        self.model = model
        self.tokenizer = tokens
        self.final_data = pd.concat(
            [self.train_df, self.dev_df, self.test_df], ignore_index=True)
        self.pred_object = pred_object

    def create_df(self, df: pd.DataFrame):
        predict_dict = {
            'actual_answer': [],
            'predicted_answer': []
        }
        print(self.final_data.shape[0])
        for i, row in tqdm(df.iterrows()):
            question = row['question']
            context = row['sentence']
            actual_answer = row['answer']
            predicted_answer = self.pred_object.make_prediction(
                question, context)

            predict_dict['actual_answer'].append(actual_answer)
            predict_dict['predicted_answer'].append(predicted_answer)

        df = pd.DataFrame(predict_dict)
        return df

    def compute_store_predictions(self):
        # Computes the predicted answer for each question and stores them.
        if not os.path.exists('data/predictions.csv'):
            df = self.create_df(self.final_data)
            df.to_csv("data/predictions.csv", index=False)

        if not os.path.exists('data/eval_predictions.csv'):
            df1 = self.create_df(self.eval_df)
            df1.to_csv("data/eval_predictions.csv", index=False)

    def precision_recall_f1(self, actual_answer: str, predicted_answer: str):
        """
        Computes token level Precision, Recall, F1-score for the complete data.

        ARGS:
            actual_answer: A string containing the actual answers.
            predicted_answer: A string containing the predicted answers.

        Returns:
            returns the token level Precision, Recall, F1. 
        """
        if type(actual_answer) == float or type(predicted_answer) == float:
            return 0, 0, 0
        actual_answer = valid_string(actual_answer)
        predicted_answer = valid_string(predicted_answer)
        actual_tokens = set(actual_answer.split())
        predicted_tokens = set(predicted_answer.split())

        # Find the common tokens between actual and predicted.
        common_tokens = actual_tokens.intersection(predicted_tokens)

        if len(predicted_tokens) == 0:
            precision = 0
        else:
            precision = len(common_tokens) / len(predicted_tokens)

        if len(actual_tokens) == 0:
            recall = 0
        else:
            recall = len(common_tokens) / len(actual_tokens)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    def display_metrics(self, dataframe: pd.DataFrame):
        """
        Takes in a dataframe of actual and predicted answers
        and returns the overall token level precsion,
        recall, f1-score.
        """
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        num_examples = len(dataframe)

        for index, row in dataframe.iterrows():
            actual_answer = row['actual_answer']
            predicted_answer = row['predicted_answer']
            p, r, f1 = self.precision_recall_f1(
                actual_answer, predicted_answer)

            total_precision += p
            total_recall += r
            total_f1 += f1

        overall_precision = total_precision / num_examples
        overall_recall = total_recall / num_examples
        overall_f1 = total_f1 / num_examples

        return overall_precision, overall_recall, overall_f1
