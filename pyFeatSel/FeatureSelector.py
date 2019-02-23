import abc
import pandas as pd
import numpy as np

from pyFeatSel.Evaluator import EvaluatorBase
from pyFeatSel.Model import Model
from pyFeatSel.Helper import create_k_fold_indices

class FeatureSelector(abc.ABC):

    def __init__(self, model: Model, train_data: pd.DataFrame, k_folds: int,
                 train_label: np.ndarray, evaluator: EvaluatorBase):
        '''
        Feature Selector Class which finds the best features for a given dataset.
        Please provide a model which must be wrapped in a class inheritaed from pyFeatSel.Model
        Please proivde an evaluator to, which must be wrapped in a class inherited from pyFeatSel.Evaluator
        :param model:
        :param train_data:
        :param k_folds:
        :param train_label:
        :param evaluator:
        '''
        assert isinstance(model, Model), "Model must be an instance ob abstract class model!"
        assert type(train_data) == pd.DataFrame, "train_data must be of type pd.DataFrame"
        assert type(train_label) == np.ndarray, "train_data must be of type pd.DataFrame"
        assert type(k_folds) == int
        assert k_folds > 0, "k_folds must be a positive integer!"
        assert k_folds < train_data.shape[0], "k_folds must be lesser than rows in train_data!"
        assert issubclass(evaluator, EvaluatorBase)

        self.model = model
        self.evaluator = evaluator
        self.train_data = train_data
        self.train_label = train_label
        self.k_fold = k_folds
        self.computed_features = []
        self.best_result = None

    def find_best_features(self):
        features = self.run_selecting()

    def inner_run(self, column_names: list):
        data = self.train_data[column_names]
        label = self.train_label
        subsets = create_k_fold_indices(label.shape[0], self.k_fold)
        measure_test = []
        measure_val = []

        for subset in subsets:
            dtrain, dtest, dval = data[subset["train"]], data[subset["test"]], data[subset["val"]]
            ltrain, ltest, lval = label[subset["train"]], label[subset["test"]], label[subset["val"]]
            self.model.train_model(dtrain, ltrain)
            y_test = self.model.predict(dtest)
            y_val = self.model.predict(dval)
            eval_test = self.evaluator(y_test, ltest)
            eval_val = self.evaluator(y_val, lval)




    @abc.abstractmethod
    def run_selecting(self):
        pass
