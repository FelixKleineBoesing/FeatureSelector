import pandas as pd
import abc
import numpy as np



class EvaluatorBase:

    def __init__(self):
        '''
        evaluator used for measuring the succes during optimization
        '''

    def evaluate(self, preds: list, actuals: list):
        '''

        :param preds:
        :param actuals:
        :return:
        '''
        assert type(preds) in [list, np.ndarray]
        assert type(actuals) in [list, np.ndarray]
        if type(preds) == list:
            preds = np.array(preds)
        if type(actuals) == list:
            actuals = np.array(actuals)
        return self._inner_evaluate(preds, actuals)

    @abc.abstractmethod
    def _inner_evaluate(self, preds: np.ndarray, actuals: np.ndarray):
        '''
        calculate a error measure
        :return: return a float that determine the error
        '''
        pass


class Precision(EvaluatorBase):

    def _inner_evaluate(self, preds: np.ndarray, actuals: np.ndarray):
        tp = preds == 1
        return sum(np.logical_and(preds == 1, actuals == 1)) / \
               (sum(np.logical_and(preds == 1, actuals == 1)) +
                sum(np.logical_and(preds == 0, actuals == 1)))


class Recall(EvaluatorBase):

    def _inner_evaluate(self, preds: np.ndarray, actuals: np.ndarray):
        return sum(np.logical_and(preds == 1, actuals == 1)) / (
                sum(np.logical_and(preds == 1, actuals == 1)) +
                sum(np.logical_and(preds == 1, actuals == 0)))


class FOneScore(EvaluatorBase):

    def _inner_evaluate(self, preds: np.ndarray, actuals: np.ndarray):
        precision = sum(np.logical_and(preds == 1, actuals == 1)) / (sum(np.logical_and(preds == 1, actuals == 1)) +
                                                                              sum(np.logical_and(preds == 0, actuals == 1)))
        recall = sum(np.logical_and(preds == 1, actuals == 1)) / (
                sum(np.logical_and(preds == 1, actuals == 1)) +
                sum(np.logical_and(preds == 1, actuals == 0)))
        return (2 * precision * recall) / (precision + recall)


class Accuracy(EvaluatorBase):

    def _inner_evaluate(self, preds: np.ndarray, actuals: np.ndarray):
        return sum(preds == actuals) / len(preds)


class RMSE(EvaluatorBase):

    def _inner_evaluate(self, preds: np.ndarray, actuals: np.ndarray):
        return np.sqrt(np.mean((actuals-preds)**2))