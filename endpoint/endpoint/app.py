from fastapi import FastAPI
from endpoint.utils.settings import Settings
from nlp_ml_api.utils.model_utils import load_model

app = FastAPI()
settings = Settings()
model = load_model('model_deploy.pickle')

@app.get("/")
async def root():
    return {"message": f"Hello World, {settings.mode}"}


@app.get("/endpoint/")
async def invoke_model(query:str=''):
    if not isinstance(query, str):
        return {"message": "Invalid Input"}
    return {"query": query,
            "prediction": "{}".format(model.predict(str(query))[0])}
