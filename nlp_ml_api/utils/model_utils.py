import pickle

from nlp_ml_api.abstractions import NLPModel
from nlp_ml_api.models import (CountModel, CountModelParams,
                               TFIDFModel, TFIDFModelParams,
                               TransformerModel, TransformerModelParams)

MODELS = {
    'CountModel': {'model': CountModel, 'params': CountModelParams},
    'TFIDFModel': {'model': TFIDFModel, 'params': TFIDFModelParams},
    'TransformerModel': {'model': TransformerModel, 'params': TransformerModelParams}
    # 'SBertModel': SBertModel,
}

SAVED_MODEL_PATH = 'saved_models'

def save_model(model: NLPModel, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename: str) -> NLPModel:
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_available_models():
    return list(MODELS.keys())
