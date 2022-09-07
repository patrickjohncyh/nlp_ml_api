import os
import subprocess
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


def deploy_model(model: NLPModel, mode='localhost'):
    assert mode in ['localhost', ]
    # save model into endpoint folder
    output_path = 'endpoint/model_deploy.pickle'
    if os.path.exists(output_path):
        os.remove(output_path)
    save_model(model, output_path)

    # call make
    p = subprocess.call(['make','-C', 'endpoint', 'prod'])
    if mode == 'localhost':
        try:
            p = subprocess.call(['make', '-C', 'endpoint', 'up-prod'])
        except KeyboardInterrupt:
            p = subprocess.call(['make', '-C', 'endpoint', 'down-prod'])