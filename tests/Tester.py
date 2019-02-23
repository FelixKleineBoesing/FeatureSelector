import unittest
import pandas as pd
import numpy as np

from pyFeatSel.Model import XGBoostModel
from pyFeatSel.CompleteFeatureSpace import CompleteFeatureSpace
from pyFeatSel.Evaluator import FOneScore


class Tester(unittest.TestCase):

    def test_run_complete_feature_space(self):
        train_data = pd.concat([pd.read_csv("../data/train_data.csv"), pd.read_csv("../data/validation_data.csv")])
        with open("../data/train_label.csv") as f:
            train_label = f.readlines()
        with open("../data/validation_label.csv") as f:
            train_label += f.readlines()

        train_label = np.array(train_label[1:])
        xgb_model = XGBoostModel(n_rounds=100, xgb_params={"eta": 0.3})
        comp_feat_selector = CompleteFeatureSpace(model=xgb_model, train_data=train_data,
                                                  train_label=train_label, k_folds=1,
                                                  evaluator=FOneScore)
        features = comp_feat_selector.run_selecting()
