"""
Ameya Santosh Gidh

This files contains the code to train and build models.
"""

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel
from utils import preprocess_data
from transformers import CONFIG_MAPPING


class Models:
    def __init__(self, train_data=None, dev_data=None, test_data=None):
        """
        Takes the training_data, dev_data, test_data into one
        final_data and trains the model.

        Args:
            train_data: training dataframe.
            dev_data: dev_dataframe.
            test_data: testing_dataframe.
        """
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        # Combine the three data frames into a single dataframe.
        self.final_data = pd.concat(
            [self.train_data, self.dev_data, self.test_data], ignore_index=True)
        self.sequence_length = 384
        # Load pretrained tokenizer from bert model.
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def prepare_data(self):
        # Prepare the x_train and y_train splits to make it ready for training.
        x_train, y_train = preprocess_data(questions=self.final_data['question'],
                                           sentences=self.final_data['sentence'],
                                           answers=self.final_data['answer'],
                                           tokenizer=self.tokenizer,
                                           seq_length=self.sequence_length)
        return x_train, y_train

    def create_qa_model(self):
        # Finetunes the Bert model for question-answering task.
        input_ids = tf.keras.layers.Input(
            shape=(self.sequence_length, ), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(
            shape=(self.sequence_length,), dtype=tf.int32, name="attention_mask")

        bert = TFBertModel.from_pretrained(
            'bert-base-uncased', return_dict=True)
        sequence_output = bert([input_ids, attention_mask])[
            'last_hidden_state']

        start_logits = tf.keras.layers.Dense(
            1, name='start_position')(sequence_output)
        # output layer to predict first_idx of answer.
        start_logits = tf.keras.layers.Flatten()(start_logits)

        end_logits = tf.keras.layers.Dense(
            1, name='end_position')(sequence_output)
        # output layer to predict last_idx of answer.
        end_logits = tf.keras.layers.Flatten()(end_logits)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=[
                               start_logits, end_logits])
        return model

    def train_model(self):
        # trains the fine-tuned model for qa_task,
        x_train, y_train = self.prepare_data()
        qa_model = self.create_qa_model()

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        qa_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        epochs = 10
        batch_size = 8

        history = qa_model.fit(x_train, [y_train['start_position'],
                                         y_train['end_position']], epochs=epochs, batch_size=batch_size)

        model_save_path = "/Users/jyothivishnuvardhankolla/Desktop/SoftinWay/Chatbot-development/Models"

        # Save the model weights
        qa_model.save_weights(model_save_path + "/tf_model.h5")

        # Save the tokenizer
        self.tokenizer.save_pretrained(model_save_path)

        # Save the config
        config = CONFIG_MAPPING["bert"]()
        config.save_pretrained(model_save_path)
