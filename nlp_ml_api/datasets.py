import os
import pandas as pd
from typing import List
from nlp_ml_api.abstractions import NLPDataset


class C19TwitterSentimentDataset(NLPDataset):
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, **kwargs):
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()


        super().__init__(**kwargs)

    def load(self):
        self._x_train = self.df_train[['OriginalTweet']]
        self._y_train = self.df_train[['Sentiment']]
        self._x_test = self.df_test[['OriginalTweet']]
        self._y_test = self.df_test[['Sentiment']]
        # set columns
        self._x_test.columns = ['text']
        self._x_train.columns = ['text']
        self._y_train.columns = ['label']
        self._y_test.columns = ['label']
        self._labels = sorted(self._y_train['label'].unique().tolist())

    @classmethod
    def from_local(cls,
                   path_to_dataset_folder: str,
                   train_set_filename: str = "Corona_NLP_train.csv",
                   test_set_filename: str = "Corona_NLP_test.csv"):
        df_train = pd.read_csv(os.path.join(path_to_dataset_folder, train_set_filename), encoding="ISO-8859-1")
        df_test = pd.read_csv(os.path.join(path_to_dataset_folder, test_set_filename), encoding="ISO-8859-1")
        return cls(df_train, df_test)

    @classmethod
    def from_list(cls, data_list_train: List[dict], data_list_test: List[dict]):
        # check input format
        assert data_list_train and all('string' in _.keys() and 'labels' in _.keys() for _ in data_list_train)
        assert data_list_test and all('string' in _.keys() and 'labels' in _.keys() for _ in data_list_test)

        df_train = pd.DataFrame(data_list_train)
        df_test = pd.DataFrame(data_list_test)
        df_train.columns = ['OriginalTweet', 'Sentiment']
        df_test.columns = ['OriginalTweet', 'Sentiment']
        return cls(df_train, df_test)
