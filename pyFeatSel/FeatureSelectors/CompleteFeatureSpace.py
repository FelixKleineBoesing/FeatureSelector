import logging

from pyFeatSel.FeatureSelectors.FeatureSelector import FeatureSelector
from pyFeatSel.misc.Helpers import create_all_combinations


class CompleteFeatureSpace(FeatureSelector):

    def run_selecting(self):
        column_names = self.train_data.columns.values.tolist()

        combns = [comb for comb in create_all_combinations(column_names) if len(comb) > 0]

        for comb in combns:
            measure = self.inner_run(list(comb))
            self.computed_features += [{"measure": measure, "column_names": comb}]
            logging.info("Test Measure: {0}, Val Measure: {1}".format(measure["test"], measure["val"]))
            if self.best_result is None:
                self.best_result = {"measure": measure, "column_names": comb}
            elif measure["test"] > self.best_result["measure"]["test"] and self.maximize_measure:
                self.best_result = {"measure": measure, "column_names": comb}
            elif measure["test"] < self.best_result["measure"]["test"] and not self.maximize_measure:
                self.best_result = {"measure": measure, "column_names": comb}
            else:
                continue
