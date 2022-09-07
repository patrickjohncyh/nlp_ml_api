from nlp_ml_api.abstractions import NLPModel
from typing import List
import gensim
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from gensim.parsing.preprocessing import remove_stopwords
import gensim.downloader as api


class W2VModel(NLPModel):
    model_name = 'W2VModel'
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.classifier = LogisticRegression(max_iter=1000)
        self.preprocessing = self.pre_processor
        self.wv = None
        self.unk = np.random.rand(100)
        self.missing = 0

    def pre_processor(self, string: str):
        return remove_stopwords(string).lower()

    def get_mean_vector(self, string:str):
        tokens = string.split(' ')
        vectors = [self.wv[_] for _ in tokens if _ in self.wv]
        if vectors:
            return np.array(vectors).mean(axis=0)
        else:
            self.missing +=1
            return self.unk

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs):
        # fit vectorizer to dataset
        sentences = [self.preprocessing(_) for _ in x_train.tolist()]
        # self.wv = train_embeddings(sentences, **kwargs)
        self.wv = api.load("glove-twitter-100")
        X_train_vect = np.stack([self.get_mean_vector(_) for _ in sentences], axis=0)
        print(self.missing)
        self.classifier.fit(X_train_vect, y_train)

    def predict(self, prediction_input: List[str], *args, **kwargs):
        prediction_input = [self.pre_processor(_) for _ in prediction_input]
        X_pred_vect = [self.get_mean_vector(_) for _ in prediction_input]
        return self.classifier.predict(X_pred_vect)


def train_embeddings(
    sentences: List[str],
    min_c: int = 5,
    size: int = 256,
    window: int = 10,
    iterations: int = 20,
    ns_exponent: float = 0.75,
    is_debug: bool = True):
    """
    :param sessions: list of lists, as user sessions are list of interactions
    :param min_c: minimum frequency of an event for it to be calculated for product embeddings
    :param size: output dimension
    :param window: window parameter for gensim word2vec
    :param iterations: number of training iterations
    :param ns_exponent: ns_exponent parameter for gensim word2vec
    :param is_debug: if true, be more verbose when training
    :return: trained product embedding model
    """
    print('Training CBOW on {} sentences'.format(len(sentences)))
    print('First sentence is "{}"'.format(sentences[0]))
    model = gensim.models.Word2Vec(sentences=[_.split(' ') for _ in sentences],
                                   min_count=min_c,
                                   vector_size=size,
                                   window=window,
                                   epochs=iterations,
                                   ns_exponent=ns_exponent)

    if is_debug:
        print("# tokens in the space: {}".format(len(model.wv.index_to_key)))

    return model.wv