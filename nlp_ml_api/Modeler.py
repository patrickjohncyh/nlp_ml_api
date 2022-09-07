import os
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from nlp_ml_api.utils.model_utils import save_model, load_model
from nlp_ml_api.abstractions import NLPDataset, NLPModelParams, NLPModel
from nlp_ml_api.utils.model_utils import MODELS, SAVED_MODEL_PATH

class Modeler:

    def __init__(self, model: [str, NLPModel], dataset: NLPDataset, model_params: NLPModelParams = None):
        self.model = None
        if isinstance(model, str):
            # create new instance of model
            if model not in MODELS:
                print("Model {} is currently not supported".format(model))
                raise NotImplementedError
            # use defaults if non provided
            params = model_params if model_params else MODELS[model]['params']()
            self.model = MODELS[model]['model'](params)
        elif isinstance(model, NLPModel):
            # load a pre-trained model
            self.model = model
        else:
            # malformed input
            raise RuntimeError("model {} is not type str or NLPModel".format(model))
        self.dataset = dataset

    def train(self, limit: int = 0, **kwargs):
        if limit:
            x_train, y_train = self.dataset.x_train.head(limit), self.dataset.y_train.head(limit)
        else:
            x_train, y_train = self.dataset.x_train, self.dataset.y_train

        self.model.fit(x_train=x_train,
                       y_train=y_train,
                       **kwargs)

    def evaluate(self, split:str = 'test'):
        assert split in ['train', 'test']
        x = self.dataset.x_test if split == 'test' else self.dataset.x_train
        y = self.dataset.y_test if split=='test' else self.dataset.y_train
        y_pred = self.model.predict(x.text)
        y_test = y
        c_rep = classification_report(y_test, y_pred, labels=self.dataset.labels)
        c_matrix = confusion_matrix(y_test, y_pred, labels=self.dataset.labels)
        # classification report
        print("============ Classification Report ============\n")
        print(c_rep)
        print("============ Error Analysis ============\n")
        # error analysis
        errors = self._error_analysis(x, y, y_pred)
        for label in self.dataset.labels:
            print('\nMis-classifications for {}:\n'.format(label))
            for _, row in errors[label][['text','pred']].sample(n=3).iterrows():
                print("Text : {}".format(row['text']))
                print("Prediction: {}\n".format(row['pred']))
            print('---')
        # confusion matrix
        sns.heatmap(c_matrix,
                    annot=True,
                    fmt='g',
                    xticklabels=self.dataset.labels,
                    yticklabels=self.dataset.labels)
        plt.xticks(rotation=30)
        plt.yticks(rotation=30)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

        return {
            'split': split,
            'classification_report': c_rep,
            'confusion_matrix': c_matrix,
            'labels': self.dataset.labels,
            'errors': errors
        }

    def save_model(self, filename: str):
        os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
        save_model(self.model, os.path.join(SAVED_MODEL_PATH, filename))

    def load_model(self, path_to_model: str):
        self.model = load_model(path_to_model)

    def _error_analysis(self, x: pd.DataFrame, y: pd.DataFrame, y_pred:List):
        y_pred = np.array(y_pred)
        prediction_errors = y.values.reshape(-1) != y_pred.reshape(-1)
        x_errors = x[prediction_errors]
        y_errors = y[prediction_errors]
        y_pred_errors = y_pred[prediction_errors]
        errors_df = pd.DataFrame(np.concatenate([
            x_errors.values,
            y_errors.values,
            y_pred_errors.reshape(-1,1)
        ], axis=1))
        errors_df.columns = ['text','label','pred']
        errors_df = errors_df.set_index('label')
        # groupby erorrs by label
        errors = {
            label : errors_df.loc[label].copy() for label in self.dataset.labels
        }
        return errors
