import pandas as pd
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertModel
import re


def find_start_end_pos(sentence: str, answer: str, tokenizer: BertTokenizer):
    """
    This function takes a sentence and answer as input and returns
    the index of first_pos and last_pos of answer in the sentence.

    Args:
        sentence: A string variable containing sentence.
        answer: A string variable containing answer.
        tokenizer: BertTokenizer function.

    Returns:
        returns the start_idx and end_idx of answer in the sentence.
    """
    sentence_tokens = tokenizer.tokenize(sentence)
    answer_tokens = tokenizer.tokenize(answer)

    # Initialize start and end_pos as -1.
    start_idx = -1
    end_idx = -1
    for i, token in enumerate(sentence_tokens):
        if token == answer_tokens[0] and sentence_tokens[i:i+len(answer_tokens)] == answer_tokens:
            start_idx = i
            end_idx = i + len(answer_tokens) - 1
            break

    return start_idx, end_idx


def preprocess_data(questions: pd.Series, sentences: pd.Series, answers: pd.Series, tokenizer: BertTokenizer, seq_length: int):
    """
    This function takes questions, sentences, answers and preprocess
    them to make ready for training the model.

    Args:
        questions: An pd.series of questions from the data.

    Returns:
        returns the input_ids, attention_mask, start_position and end_position for
        training the model.
    """
    input_ids, attention_masks, start_positions, end_positions = [], [], [], []
    for question, sentence, answer in zip(questions, sentences, answers):
        # Encode the words.
        encoder = tokenizer(question, sentence, padding='max_length',
                            truncation=True, max_length=seq_length)
        # Find the start and end_idx of answer.
        start_pos, end_pos = find_start_end_pos(sentence, answer, tokenizer)

        # Ignore rows where answer is not in the sentence.
        if start_pos == -1 and end_pos == -1:
            continue

        input_ids.append(encoder['input_ids'])
        attention_masks.append(encoder['attention_mask'])
        start_positions.append(start_pos)
        end_positions.append(end_pos)

    return {
        'input_ids': tf.constant(input_ids, dtype=tf.int32),
        'attention_mask': tf.constant(attention_masks, dtype=tf.int32)
    }, {
        'start_position': tf.constant(start_positions, dtype=tf.int32),
        'end_position': tf.constant(end_positions, dtype=tf.int32)
    }

# Load the model architecture


def create_qa_model(config):
    sequence_length = 384
    input_ids = tf.keras.layers.Input(
        shape=(sequence_length,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(
        shape=(sequence_length,), dtype=tf.int32, name='attention_mask')

    bert = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    sequence_output = bert([input_ids, attention_mask])[0]

    start_logits = tf.keras.layers.Dense(
        1, name='start_position')(sequence_output)
    start_logits = tf.keras.layers.Flatten()(start_logits)

    end_logits = tf.keras.layers.Dense(1, name='end_position')(sequence_output)
    end_logits = tf.keras.layers.Flatten()(end_logits)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=[
                           start_logits, end_logits])
    return model


def load_model(model_path: str):
    # Load the config.
    config = BertConfig.from_pretrained(model_path)

    # Load the model.
    qa_model = create_qa_model(config)
    qa_model.load_weights(model_path + "/tf_model.h5")
    return qa_model

def valid_string(string: str):
    words = re.findall(r'\b\w+\b', string)
    output_string = ' '.join(words)
    return output_string
