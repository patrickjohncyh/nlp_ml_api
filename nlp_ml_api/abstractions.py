from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union


@dataclass
class NLPModelParams:
    pass


class NLPModel(ABC):
    """
    Abstract class for recommendation model
    """

    def __init__(self,  **kwargs):
        """
        :param model: a model that can be used in the predict function
        """
        pass

    @abstractmethod
    def predict(self, prediction_input: Union[str, List[str]], *args, **kwargs):
        """
        The predict function should implement the behaviour of the model at inference time.
        :param prediction_input: the input that is used to to do the prediction
        :param args:
        :param kwargs:
        :return:
        """
        return NotImplementedError

    @abstractmethod
    def fit(self, **kwargs):
        return NotImplementedError


class NLPDataset(ABC):

    def __init__(self, force_download=False):
        """
        :param force_download: allows to force the download of the dataset in case it is needed.
        :type: force_download: bool, optional
        """
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self._labels = None
        self.force_download = force_download
        self.load()

    @abstractmethod
    def load(self, **kwargs):
        """
        Abstract method that should implement dataset loading
        :return:
        """
        return

    @property
    def labels(self):
        return self._labels

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_test(self):
        return self._y_test
