from nlp_ml_api.abstractions import NLPModel, NLPModelParams
from nlp_ml_api.utils.TweetNormalizer import normalizeTweet
from typing import List, Union
import torch
import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass(init=True)
class TransformerModelParams(NLPModelParams):
    model_type: str = "roberta"
    model_name: str = "vinai/bertweet-covid19-base-uncased"
    tokenizer_name: str = "vinai/bertweet-covid19-base-uncased"
    epochs: int = 1
    preprocessor = staticmethod(normalizeTweet)
    evaluate_during_training: bool = True
    evaluate_during_training_verbose: bool = True


class TransformerModel(NLPModel):
    model_name = 'TransformerModel'

    def __init__(self, model_params: TransformerModelParams,  **kwargs):
        super().__init__(**kwargs)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print('USING DEVICE : {}'.format(self.device))
        self.model_params = model_params
        self.binarizer = LabelBinarizer()
        # Optional model configuration
        model_args = ClassificationArgs(num_train_epochs=model_params.epochs,
                                        overwrite_output_dir=True,
                                        evaluate_during_training=model_params.evaluate_during_training,
                                        evaluate_during_training_verbose=model_params.evaluate_during_training_verbose)

        self.preprocessor = model_params.preprocessor

        # Create a ClassificationModel
        self.classifier = ClassificationModel(
            model_params.model_type,
            model_params.model_name,
            tokenizer_type=AutoTokenizer.from_pretrained(model_params.tokenizer_name, use_fast=True),
            args=model_args,
            use_cuda=(self.device == 'cuda:0'),
            num_labels=5
        )

    def build_input_df(self, x: pd.DataFrame, y: pd.DataFrame):
        # fit binarizer; convert to one-hot
        y = self.binarizer.fit_transform(y.values)
        # converse to sparse categorical
        y = pd.DataFrame(np.argwhere(y)[:, 1])
        # build input df w correct labels
        df = pd.concat([x.reset_index(drop=True), y], axis=1, ignore_index=True)
        df.columns = ["text", "labels"]
        return df

    def fit(self, x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            train_size: float = 0.9,
            random_state: int = None,
            shuffle: bool = True,
            sample_weight=None,
            verbose=True,
            **kwargs):

        # keep some data for validation
        _x_train, _x_val, _y_train, _y_val = train_test_split(x_train,
                                                              y_train,
                                                              train_size=train_size,
                                                              random_state=random_state,
                                                              shuffle=shuffle)
        # build input df for training
        train_df = self.build_input_df(_x_train.text.apply(self.preprocessor).to_frame(),
                                       _y_train)
        eval_df = self.build_input_df(_x_val.text.apply(self.preprocessor).to_frame(),
                                      _y_val)
        print(train_df.head(5))
        # train model one all data
        self.classifier.train_model(train_df, eval_df=eval_df, show_running_loss=verbose)

        # get model performance
        _y_train_pred = self.predict(_x_train.text.tolist())
        _y_val_pred = self.predict(_x_val.text.tolist())
        if verbose:
            print("Train Accuracy : {}".format(accuracy_score(_y_train.values, _y_train_pred)))
            print("Train Accuracy : {}".format(accuracy_score(_y_val.values, _y_val_pred)))

    def predict(self, prediction_input: Union[str, List[str]], *args, **kwargs):
        if isinstance(prediction_input, str):
            prediction_input = [prediction_input]
        prediction_input = [self.preprocessor(_) for _ in prediction_input]
        pred = self.classifier.predict(prediction_input)
        # convert prediction back to original class labels
        return [self.binarizer.classes_[_] for _ in pred[0]]
