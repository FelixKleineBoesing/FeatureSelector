import pandas as pd
import abc
import numpy as np
from sklearn.metrics import confusion_matrix


class EvaluatorBase:

    def __init__(self, preds: list, actuals: list):
        assert type(preds) in [list, np.ndarray]
        assert type(actuals) in [list, np.ndarray]
        if type(preds) == list:
            self.pred = np.array(preds)
        else:
            self.pred = preds
        if type(actuals) == list:
            self.actuals = np.array(actuals)
        else:
            self.actuals = actuals

    @abc.abstractmethod
    def evaluate(self):
        '''
        calculate a error measure
        :return: return a float that determine the error
        '''
        pass


class Precision(EvaluatorBase):

    def evaluate(self):
        tp = self.pred == 1
        return sum(np.logical_and(self.pred == 1, self.actuals == 1)) / \
               (sum(np.logical_and(self.pred == 1, self.actuals == 1)) +
                sum(np.logical_and(self.pred == 0, self.actuals == 1)))


class Recall(EvaluatorBase):

    def evaluate(self):
        return sum(np.logical_and(self.pred == 1, self.actuals == 1)) / (
                sum(np.logical_and(self.pred == 1, self.actuals == 1)) +
                sum(np.logical_and(self.pred == 1, self.actuals == 0)))


class FOneScore(EvaluatorBase):

    def evaluate(self):
        precision = sum(np.logical_and(self.pred == 1, self.actuals == 1)) / (sum(np.logical_and(self.pred == 1, self.actuals == 1)) +
                                                                              sum(np.logical_and(self.pred == 0, self.actuals == 1)))
        recall = sum(np.logical_and(self.pred == 1, self.actuals == 1)) / (
                sum(np.logical_and(self.pred == 1, self.actuals == 1)) +
                sum(np.logical_and(self.pred == 1, self.actuals == 0)))
        self.f_one_score = (2 * precision * recall) / (precision + recall)


class Accuracy(EvaluatorBase):

    def evaluate(self):
        return sum(self.pred == self.actuals) / len(self.pred)
