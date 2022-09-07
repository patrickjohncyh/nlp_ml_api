import os
from typing import List
import pandas as pd
import numpy as np
import torch

from nlp_ml_api.abstractions import NLPModel
from nlp_ml_api.utils.TweetNormalizer import normalizeTweet
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier

class SBertModel(NLPModel):
    model_name = 'SBertModel'
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = normalizeTweet
        self.classifier = MLPClassifier(max_iter=10000,
                                        hidden_layer_sizes=(512,512),
                                        validation_fraction=0.2,
                                        activation='relu',
                                        early_stopping=True,
                                        n_iter_no_change=20,
                                        learning_rate_init=1e-4,
                                        batch_size=128
                                        )
        self.pre_trained_model_name = 'all-mpnet-base-v2'
        self.model = SentenceTransformer(self.pre_trained_model_name)

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs):
        sentences = [self.preprocessor(_) for _ in x_train.tolist()]
        vector_path = os.path.join('data',self.pre_trained_model_name+'-corona-nlp-train.npy')
        if os.path.exists(vector_path):
            print('USING PRE-GENERATED VECTORS!')
            X_train_vect = np.load(vector_path)
        else:
            with torch.no_grad():
                X_train_vect = self.model.encode(sentences, show_progress_bar=True)
                np.save(vector_path, X_train_vect)
        self.classifier.fit(X_train_vect, y_train)
        print('Train Accuracy : {}'.format(self.classifier.score(X_train_vect, y_train)))

    def predict(self, prediction_input: List[str], *args, **kwargs):
        vector_path = os.path.join('data', self.pre_trained_model_name + '-corona-nlp-test.npy')
        if os.path.exists(vector_path):
            print('USING PRE-GENERATED VECTORS!')
            X_pred_vect = np.load(vector_path)
        else:
            with torch.no_grad():
                X_pred_vect = self.model.encode([normalizeTweet(_) for _ in prediction_input], show_progress_bar=True)
                np.save(vector_path, X_pred_vect)
        return self.classifier.predict(X_pred_vect)
