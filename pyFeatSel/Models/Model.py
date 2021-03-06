import abc
import pandas as pd
import xgboost as xgb
import numpy as np
import logging


class Model(abc.ABC):

    @abc.abstractmethod
    def train_model(self, train_data: pd.DataFrame, train_label: list):
        pass

    @abc.abstractmethod
    def predict(self, test_data: pd.DataFrame):
        pass

    @staticmethod
    def preprocess(train_data: pd.DataFrame):
        pass


class XGBoostModel(Model):

    def __init__(self, xgb_params: {}, n_rounds: int):
        assert type(xgb_params) == dict, "xgb_params is of type list"
        assert type(n_rounds) == int, "n_rounds must be of type int"
        assert n_rounds > 1, "nrounds must be greater than one"
        if len(xgb_params) == 0:
            logging.info("No params are delivered. We´ll default xgb params to 0.3 eta")
            self.params = {"eta": 0.3}
        else:
            self.params = xgb_params
        self.params["silent"] = 1
        self.model = None
        self.n_rounds = n_rounds

    def train_model(self, train_data: pd.DataFrame, train_label: np.ndarray):
        assert type(train_data) == pd.DataFrame
        assert type(train_label) == np.ndarray
        train_data = self._preprocess(train_data)
        dtrain = xgb.DMatrix(train_data, train_label, silent=True)

        self.model = xgb.train(self.params, dtrain, num_boost_round=self.n_rounds, verbose_eval=False)

    def predict(self, data: pd.DataFrame):
        data = self._preprocess(data)
        dtest = xgb.DMatrix(data)
        y_pred = self.model.predict(dtest)
        return y_pred

    @staticmethod
    def _preprocess(data: pd.DataFrame):
        # one hot encode
        data = pd.get_dummies(data)

        return data
