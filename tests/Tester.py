import pandas as pd
import numpy as np
import logging
import time

from pyFeatSel.Models.Model import XGBoostModel
from pyFeatSel.FeatureSelectors.CompleteFeatureSpace import CompleteFeatureSpace
from pyFeatSel.FeatureSelectors.GreedySearch import GreedySearch
from pyFeatSel.Evaluator.Evaluator import Accuracy
from pyFeatSel.misc.Helpers import threshold_base


class Tester:

    def run_complete_feature_space(self):
        start = time.time()
        logging.getLogger().setLevel(logging.INFO)
        train_data, train_label = self.read_files()

        xgb_model = XGBoostModel(n_rounds=100, xgb_params={"eta": 0.3})
        comp_feat_selector = CompleteFeatureSpace(model=xgb_model, train_data=train_data,
                                                  train_label=train_label, k_folds=5,
                                                  evaluator=Accuracy(), maximize_measure=True,
                                                  objective="classification", threshold_func=threshold_base)
        comp_feat_selector.run_selecting()
        logging.info("Used time in seconds: {0}, got test (val) measure: {1} ({2})".
                     format(str(int(time.time()-start)),comp_feat_selector.best_result["measure"]["test"],
                            comp_feat_selector.best_result["measure"]["val"]))

    def run_greedy_search(self):
        start = time.time()
        logging.getLogger().setLevel(logging.INFO)
        train_data, train_label = self.read_files()

        xgb_model = XGBoostModel(n_rounds=100, xgb_params={"eta": 0.3})
        comp_feat_selector = GreedySearch(model=xgb_model, train_data=train_data,
                                          train_label=train_label, k_folds=10,
                                          evaluator=Accuracy(), maximize_measure=True,
                                          objective="classification", threshold_func=threshold_base)
        comp_feat_selector.run_selecting()
        logging.info(
            "Used time in seconds: {0}, got test (val) measure: {1} ({2})".format(str(int(time.time() - start)),
                                                                                  comp_feat_selector.best_result[
                                                                                      "measure"]["test"],
                                                                                  comp_feat_selector.best_result[
                                                                                      "measure"]["val"]))

    def run_greedy_search2(self):
        start = time.time()
        logging.getLogger().setLevel(logging.INFO)
        train_data, train_label = self.read_files()

        comp_feat_selector = GreedySearch(train_data=train_data, train_label=train_label, k_folds=10,
                                          objective="classification")
        comp_feat_selector.run_selecting()
        logging.info(
            "Used time in seconds: {0}, got test (val) measure: {1} ({2})".format(str(int(time.time() - start)),
                                                                                  comp_feat_selector.best_result[
                                                                                      "measure"]["test"],
                                                                                  comp_feat_selector.best_result[
                                                                                      "measure"]["val"]))


    def read_files(self):
        train_data = pd.concat([pd.read_csv("../data/train_data.csv"), pd.read_csv("../data/validation_data.csv")])
        with open("../data/train_label.csv") as f:
            train_label = f.readlines()[1:]
        with open("../data/validation_label.csv") as f:
            train_label += f.readlines()[1:]

        train_label = np.char.replace(np.array(train_label), "\n", "").astype(int)
        return train_data, train_label

if __name__=="__main__":
    tester = Tester()
    #tester.run_complete_feature_space()
    tester.run_greedy_search()
    tester.run_greedy_search2()