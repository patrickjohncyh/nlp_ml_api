# NLP ML API


This library provides utility for the training and evaluation of NLP models. It provides
several abstractions to work with namely `NLPModel` and `NLPDataset`, that would (in theory)
allow for training and evaluation of many models across various datasets.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patrickjohncyh/nlp_ml_api/blob/master/api_tutorial.ipynb)

## Installation


1. Clone the repo locally
    ```
    $ git clone https://github.com/patrickjohncyh/nlp_ml_api.git
    ```
2. Navigate to repository
   ```
    $ cd nlp_ml_api
    ```
3. Install using pip:
    ```
    $ pip install .
    ```


## Quickstart (How To Use)

```python
from nlp_ml_api.datasets import C19TwitterSentimentDataset
from nlp_ml_api.Modeler import Modeler

# load dataset
dataset = C19TwitterSentimentDataset.from_local('datasets/corona_nlp')
# load Modeler with model and dataset
modeler = Modeler('CountModel', dataset)
# train 
modeler.train()
# evaluate on test data
modeler.evaluate(split='test')
# get model from Modeler
model = modeler.model
# invoke Model locally
model.predict('Quarantine Day 5: I forgot how the outside world looks like anymore')
```

## Deployment

You can invoke the model either in python (e.g. notebook) or locally (localhost) or deploy the model
as a [Heroku](https://www.heroku.com) app and make a HTTP request at the endpoint.

### 1. Python Invocation
```python
model.predict('I like cats')
```

### 2. Localhost

We build a docker container, containing a FastAPI app which loads the model and handles requests.

```python
deploy_model(model, mode='localhost')
```

### 3. Live Endpoint / Heroku

Deployment to Heroku is more involved and assumes that you have [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
installed, and have logged in (i.e. `heroku login`). Once that is done, you can deploy the model
via `deploy_model` function. Under the hood, python issues commands via a `Makefile`.

```python
deploy_model(model, mode='heroku', endpoint_name='tweet_sentiment_nlp_endpoint')
```


## Implementation Details

### Abstractions
At a high level there are two main abstractions: 

### `NLPModel`

This abstraction implements a type of classification model. To create a new model,
you should inherit this class and implement `predict` and `fit` functions. We provide
several models out of the box:

1. `CountModel`: Classification model based on token counts and logistic regression
2. `TFIDFModel`: Classification model based on tf-idf  and logistic regression
3. `TransformerModel`: Transformer based model (Trainable on Colab!) 

Optionally, each `NLPModel` has an `NLPModelParams` dataclass which allows for changing
of hyper-parameters.

### `NLPDataset`

This abstraction wraps a dataset and provides a unified interface for models to interact with.
We provide [`C19TwitterSentimentDataset`](https://www.kaggle.com/code/kerneler/starter-covid-19-nlp-text-d3a3baa6-e) 
dataset out of the box.

### Training and Evaluation

`Modeler` takes in an`NLPDataset` and an `NLPModel`, and provides functionality
to `train` and `evaluate` the model on given dataset. It also provides utility to save
and load a model after training.
 

## Improvements / To Do

- __Metaflow/Ray__: While local machines may not have GPUs locally, [Metaflow](https://metaflow.org/) /
  [Anyscale(Ray)](https://www.anyscale.com/) can provide preemptable GPU resources specifically
  for the training step for large models (e.g. Transformers).
- __Deployment__: Deployment currently relies on model being `git push-able`, which is impractical for
  models which large memory footprint.
- __Enable custom metrics for evaluation__: Evaluation is hard-coded, but could potentially allow
  uers to pass in their own evaluation functions if required.
- Improve abstractions to suit NLP use-case: There are often standard steps in NLP (e.g. pre-processing) which
  can be made more explicit as part of the `NLPModel` abstraction.
- Improve debugging / reporting


















