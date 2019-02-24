import abc
import pandas as pd
import numpy as np
import logging
import types

from pyFeatSel.Evaluator.Evaluator import EvaluatorBase, RMSE, Accuracy
from pyFeatSel.Models.Model import Model
from pyFeatSel.misc.Helpers import create_k_fold_indices, threshold_base
from pyFeatSel.Models.Model import XGBoostModel


class FeatureSelector(abc.ABC):

    def __init__(self, train_data: pd.DataFrame, train_label: np.ndarray,objective: str = "classification",
                 k_folds: int = 1, evaluator: EvaluatorBase = None, model: Model = None,
                 maximize_measure: bool = None, threshold_func: types.FunctionType = None):
        '''
        Feature Selector Class which finds the best features for a given dataset.
        Please provide a model which must be wrapped in a class inheritaed from pyFeatSel.Model
        Please proivde an evaluator to, which must be wrapped in a class inherited from pyFeatSel.Evaluator
        :param model: instance of Model with methods for train and predict
        :param train_data:
        :param k_folds:
        :param train_label:
        :param evaluator:
        :param maximize_measure: If True measure should be maximized, of not must be minimized
        :param objective "classification" or "regression"
        :param threshold_func If objective == classification, than a threshold function is neccessary, which takes the
        predictions and maps the probability to 1/0 classes. If now function is delivered, than threshold will be set to
        0.5
        '''
        if model is not None:
            assert isinstance(model, Model), "Model must be an instance ob abstract class model!"
        else:
            logging.info("No Model delivered! Will be deafulted to XGBoost with 100 n_rounds an 0.3 learning rate")
            xgb_params = {"eta": 0.3, "objective": "reg:linear" if objective == "regresssion" else "reg:logistic"}
            model = XGBoostModel(n_rounds=100, xgb_params=xgb_params)
        assert type(train_data) == pd.DataFrame, "train_data must be of type pd.DataFrame"
        assert type(train_label) == np.ndarray, "train_data must be of type list"
        assert type(k_folds) == int
        assert k_folds > 0, "k_folds must be a positive integer!"
        assert k_folds < train_data.shape[0], "k_folds must be lesser than rows in train_data!"
        if evaluator is not None:
            assert isinstance(evaluator, EvaluatorBase)
        else:
            logging.info("No evaluator delivered! Will be defaulted to rmse for regression and "
                         "accuracy for classification")
            evaluator = Accuracy() if objective == "classification" else RMSE()
        if maximize_measure is not None:
            assert type(maximize_measure) == bool
        else:
            logging.debug("accuracy will be maximized, rmse will be minimized")
            maximize_measure = True if objective == "classification" else False
        assert type(objective) == str
        assert objective in ["classification", "regression"]
        if objective == "classification":
            if threshold_func is not None:
                assert type(threshold_func) == types.FunctionType
            else:
                logging.info("Threshold function will be defaulted to 0.5 threshold")
                threshold_func = threshold_base
        self.model = model
        self.evaluator = evaluator
        self.train_data = train_data
        self.train_label = train_label
        self.k_fold = k_folds
        self.maximize_measure = maximize_measure
        self.objective = objective
        self.threshold_func = threshold_func
        self.computed_features = []
        self.best_result = None

    def inner_run(self, column_names: list):
        data = self.train_data[column_names]
        label = self.train_label
        subsets = create_k_fold_indices(label.__len__(), self.k_fold)
        measure_test = []
        measure_val = []

        for subset in subsets:
            dtrain, dtest, dval = data.iloc[subset["train"]], data.iloc[subset["test"]], data.iloc[subset["val"]]
            ltrain, ltest, lval = label[subset["train"]], label[subset["test"]], label[subset["val"]]
            self.model.train_model(dtrain, ltrain)
            y_test = self.model.predict(dtest)
            y_val = self.model.predict(dval)
            if self.objective == "classification":
                y_test = self.threshold_func(y_test)
                y_val = self.threshold_func(y_val)
            measure_test += [self.evaluator.evaluate(y_test, ltest)]
            measure_val += [self.evaluator.evaluate(y_val, lval)]
            mean_test = np.mean(measure_test)
            mean_val = np.mean(measure_val)

        return {"test": mean_test, "val": mean_val}

    @abc.abstractmethod
    def run_selecting(self):
        pass
