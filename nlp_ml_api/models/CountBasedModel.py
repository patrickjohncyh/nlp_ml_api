from nlp_ml_api.abstractions import NLPModel, NLPModelParams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from nlp_ml_api.utils.TweetNormalizer import normalizeTweet
from sklearn.linear_model import LogisticRegression
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from typing import List, Union
from sklearn.metrics import accuracy_score


def get_train_val_split(self,
                        test_size: float = None,
                        train_size: float = None,
                        random_state: int = None,
                        shuffle: bool = True,
                        stratify=None) -> List[pd.DataFrame]:
    return train_test_split(self._x_train,
                            self.y_train,
                            test_size=test_size,
                            train_size=train_size,
                            random_state=random_state,
                            shuffle=shuffle,
                            stratify=stratify)


@dataclass(init=True)
class CountModelParams(NLPModelParams):
    ngram_range: tuple = (1, 2)
    min_df: int = 5
    stop_words: str = 'english'
    max_iter: int = 5000
    feat_sel_score_fn = staticmethod(chi2)
    feat_sel_num_feat: int = 2048
    preprocessor = staticmethod(normalizeTweet)


class CountModel(NLPModel):
    model_name = 'CountModel'

    def __init__(self, model_params: CountModelParams, **kwargs):
        super().__init__(**kwargs)

        self.model_params = model_params
        self.vectorizer = CountVectorizer(ngram_range=model_params.ngram_range,
                                          min_df=model_params.min_df,
                                          stop_words=model_params.stop_words)
        self.classifier = LogisticRegression(max_iter=model_params.max_iter)
        self.feat_selector = SelectKBest(score_func=model_params.feat_sel_score_fn,
                                         k=model_params.feat_sel_num_feat) if model_params.feat_sel_score_fn else None
        self.preprocessor = model_params.preprocessor

    def fit(self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            train_size: float = 0.8,
            random_state: int = None,
            shuffle: bool = True,
            sample_weight=None,
            verbose: bool = True,
            **kwargs):

        _x_train, _x_val, _y_train, _y_val = train_test_split(x_train,
                                                              y_train,
                                                              train_size=train_size,
                                                              random_state=random_state,
                                                              shuffle=shuffle)
        # fit vectorizer to dataset
        _x_train_inp = _x_train['text'].apply(self.preprocessor).tolist() \
            if self.preprocessor else _x_train['text'].tolist()
        _X_train_vect = self.vectorizer.fit_transform(_x_train_inp)
        if self.feat_selector:
            _X_train_vect = self.feat_selector.fit_transform(_X_train_vect, _y_train['label'])
        self.classifier.fit(_X_train_vect, _y_train['label'], sample_weight)
        if verbose:
            print('Train Accuracy: {}'.format(self.classifier.score(_X_train_vect, _y_train['label'])))
            print('Val Accuracy: {}'.format(accuracy_score(self.predict(_x_val['text'].tolist()), _y_val['label'])))

    def predict(self, prediction_input: Union[str, List[str]], *args, **kwargs):
        if isinstance(prediction_input, str):
            prediction_input = [prediction_input]
        X_pred_vect = self.vectorizer.transform([self.preprocessor(_) for _ in prediction_input])
        if self.feat_selector:
            X_pred_vect = self.feat_selector.transform(X_pred_vect)
        return self.classifier.predict(X_pred_vect)

    # def save_model(self):


@dataclass(init=True)
class TFIDFModelParams(NLPModelParams):
    ngram_range: tuple = (1, 2)
    min_df: int = 5
    stop_words: str = 'english'
    max_iter: int = 5000
    feat_sel_score_fn = staticmethod(chi2)
    feat_sel_num_feat: int = 2048
    preprocessor = staticmethod(normalizeTweet)


class TFIDFModel(NLPModel):
    model_name = 'TFIDFModel'

    def __init__(self, model_params: TFIDFModelParams, **kwargs):
        super().__init__(**kwargs)
        self.model_params = model_params
        self.vectorizer = TfidfVectorizer(ngram_range=model_params.ngram_range,
                                          min_df=model_params.min_df,
                                          stop_words=model_params.stop_words)
        self.classifier = LogisticRegression(max_iter=model_params.max_iter)
        self.feat_selector = SelectKBest(score_func=model_params.feat_sel_score_fn,
                                         k=model_params.feat_sel_num_feat) if model_params.feat_sel_score_fn else None
        self.preprocessor = model_params.preprocessor

    def fit(self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            train_size: float = 0.8,
            random_state: int = None,
            shuffle: bool = True,
            sample_weight=None,
            verbose=True,
            **kwargs):

        _x_train, _x_val, _y_train, _y_val = train_test_split(x_train,
                                                              y_train,
                                                              train_size=train_size,
                                                              random_state=random_state,
                                                              shuffle=shuffle)
        # fit vectorizer to dataset
        _x_train_inp = _x_train['text'].apply(self.preprocessor).tolist() \
                                       if self.preprocessor else _x_train['text'].tolist()
        _X_train_vect = self.vectorizer.fit_transform(_x_train_inp)
        if self.feat_selector:
            _X_train_vect = self.feat_selector.fit_transform(_X_train_vect, _y_train['label'])
        self.classifier.fit(_X_train_vect, _y_train['label'], sample_weight)
        if verbose:
            print('Train Accuracy: {}'.format(self.classifier.score(_X_train_vect, _y_train['label'])))
            print('Val Accuracy: {}'.format(accuracy_score(self.predict(_x_val['text'].tolist()), _y_val['label'])))

    def predict(self, prediction_input: List[str], *args, **kwargs):
        if isinstance(prediction_input, str):
            prediction_input = [prediction_input]
        X_pred_vect = self.vectorizer.transform([self.preprocessor(_) for _ in prediction_input])
        if self.feat_selector:
            X_pred_vect = self.feat_selector.transform(X_pred_vect)
        return self.classifier.predict(X_pred_vect)
