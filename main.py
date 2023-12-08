"""
This is the main file which performs tasks such as
explorataroy data anaysis,data-preparation,model-training based on command line inputs.
"""
import sys
from Exploratory_data_analysis import ExploratoryAnalysis
from Data_prep import PrepareData
from models import Models
import pandas as pd
from transformers import BertConfig, BertTokenizer
from utils import load_model
from Predictions import Predictions, Evaluations


def main(argv):
    """
        Main which takes command line arguements and executes the pipeline.

        ARGS:
            argv[1]: if given-1 performs exploratory data analysis on given data.
            argv[2]: if given-1 performing preprocessing and prepares the final data.
            argv[3]: if given-1 performs the training.
            argv[4]: if given-1 performs predictions based on given inputs.
            argv[5]: if given-1 performs evaluations and displays them.
    """
    perform_eda = int(argv[1])
    prepare_data = int(argv[2])
    train_mode = int(argv[3])
    prediction_mode = int(argv[4])
    evaluation_mode = int(argv[5])

    # Paths to data.
    train_tsv_path = "data/train.csv"
    dev_tsv_path = "data/dev.csv"
    test_tsv_path = "data/test.csv"

    if perform_eda == 1:  # If given 1 by user perfrom EDA.
        # Initialize the ExploratoryAnalysis object.
        ob = ExploratoryAnalysis(train_tsv=train_tsv_path, dev_tsv=dev_tsv_path,
                                 test_tsv=test_tsv_path)

        ob.exploreTrainTsv()  # Performs EDA for TrainTSV.
        ob.exploreDevTsv()  # Performs EDA for DevTSV.
        ob.exploreTestTsv()  # Performs EDA for TestTSV

    if prepare_data == 1:  # If given 1 by user prepares the data need to train the model.
        pos_ans_path = "data/WikiQASent.pos.ans.tsv"
        # Initialize the PrepareData object.
        ob = PrepareData(train_tsv=train_tsv_path, dev_tsv=dev_tsv_path,
                         test_tsv=test_tsv_path, pos_ans_tsv=pos_ans_path)
        # Preprocess the data and prepares the final data that is ready for training.
        ob.Preprocess()

    # Note: Run prepare data atleast once before running for training data to be available.
    if train_mode == 1:  # If given 1 by user the model training begins.
        train_df = pd.read_csv("data/train.csv")  # Load the train_df.
        dev_df = pd.read_csv("data/dev.csv")  # Load the dev_df.
        test_df = pd.read_csv("data/test.csv")  # Load the test_df
        ob = Models(train_df, dev_df, test_df)
        ob.train_model()  # Train the model.

    # Note: Run train_model atleast once before you run this.
    if prediction_mode == 1:  # If given 1 by user prediction mode turns on.
        loaded_model = load_model("Models")
        # Load tokenizer from fine-tuned model.
        tokenizer = BertTokenizer.from_pretrained("Models")
        # Initialize the Predictions object.
        ob = Predictions(loaded_model, tokenizer)
        question = "what is carnot cycle"  # Input question
        context = "bkfdjbvkbvkhbvhkdfbvakbhvbkd fbk"
        # Make a prediction using the loaded model and tokenizer
        answer = ob.make_prediction(question, context)
        print("Answer:", answer)

    # Note: Run train_model atleast once before you run this.
    if evaluation_mode == 1:  # If given 1 by user computes and displays the metrics.
        loaded_model = load_model("Models")
        tokenizer = BertTokenizer.from_pretrained("Models")
        train_df = pd.read_csv("data/train.csv")
        dev_df = pd.read_csv("data/dev.csv")
        test_df = pd.read_csv("data/test.csv")
        eval_df = pd.read_csv("data/eval.csv")
        ob1 = Predictions(loaded_model, tokenizer)  # Predictions object.
        # Evaluations object.
        ob2 = Evaluations(train_df, dev_df, test_df, eval_df,
                          loaded_model, tokenizer, ob1)
        # Compute predictions and store them in a csv_file.
        ob2.compute_store_predictions()
        pr, re, f1 = ob2.display_metrics(pd.read_csv("data/predictions.csv"))
        print(f"train data:Precision is {pr} and recall is {re} and f1-score is {f1}")
        pr, re, f1 = ob2.display_metrics(pd.read_csv("data/eval_predictions.csv"))
        print(f"test data:Precision is {pr} and recall is {re} and f1-score is {f1}")


if __name__ == "__main__":
    main(sys.argv)
