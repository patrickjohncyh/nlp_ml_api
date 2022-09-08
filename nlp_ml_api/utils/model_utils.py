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


def deploy_model(model: NLPModel, mode:str = 'localhost', endpoint_name:str = None):
    assert mode in ['localhost', 'heroku']

    # save model into endpoint folder
    output_path = 'endpoint/model_deploy.pickle'
    if os.path.exists(output_path):
        os.remove(output_path)
    save_model(model, output_path)

    if mode == 'localhost':
        # call make
        p = subprocess.call(['make','-C', 'endpoint', 'prod'])
        if mode == 'localhost':
            try:
                p = subprocess.check_output(['make', '-C', 'endpoint', 'up-prod'])
                print(p)
            except KeyboardInterrupt:
                p = subprocess.call(['make', '-C', 'endpoint', 'down-prod'])
            except Exception as e:
                print(e)
                p = subprocess.call(['make', '-C', 'endpoint', 'down-prod'])

    elif mode == 'heroku':
        assert endpoint_name
        local_env = os.environ.copy()
        local_env['HEROKU_APP_NAME'] = endpoint_name
        # create the heroku app
        p = subprocess.call(['make', '-C', 'endpoint', 'create_app'], env=local_env)
        # deploy app
        p = subprocess.call(['make', '-C', 'endpoint', 'deploy'], env=local_env)

    else:
        raise NotImplementedError