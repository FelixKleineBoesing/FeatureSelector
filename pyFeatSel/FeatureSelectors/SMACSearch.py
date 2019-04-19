import logging
from pyFeatSel.FeatureSelectors.FeatureSelector import FeatureSelector


class SMACSearch(FeatureSelector):

    def run_selecting(self):
        column_names = self.train_data.columns.values.tolist()
        chosen_column_names = []
        early_stopping = False

        while len(column_names) > 0 and not early_stopping:
            result = self._inner_loop(column_names, chosen_column_names)
            if self.best_result is not None:
                if result["measure"]["test"] < self.best_result["measure"]["test"]:
                    if self.maximize_measure:
                        early_stopping = True
                    else:
                        self.best_result = result
                if result["measure"]["test"] > self.best_result["measure"]["test"]:
                    if not self.maximize_measure:
                        early_stopping = True
                    else:
                        self.best_result = result
            else:
                self.best_result = result
            if not early_stopping:
                chosen_column_names += [column_names.pop(result["col_id"])]
            logging.info("Best solution! Test Measure: {0}, Val Measure: {1}".format(self.best_result["measure"]["test"],
                                                                                     self.best_result["measure"]["val"]))

    def _inner_loop(self, column_names: list, chosen_column_names: list):
        best_result = None
        for i, column_name in enumerate(column_names):
            col_names = [column_name] + chosen_column_names
            measure = self.inner_run(col_names)
            self.computed_features += [{"measure": measure, "column_names": col_names}]
            logging.debug("Test Measure: {0}, Val Measure: {1}".format(measure["test"], measure["val"]))
            if best_result is None:
                best_result = {"measure": measure, "column_names": col_names, "new_column": column_name, "col_id": i}
            elif measure["test"] > best_result["measure"]["test"] and self.maximize_measure:
                best_result = {"measure": measure, "column_names": col_names, "new_column": column_name, "col_id": i}
            elif measure["test"] < best_result["measure"]["test"] and not self.maximize_measure:
                best_result = {"measure": measure, "column_names": col_names, "new_column": column_name, "col_id": i}
            else:
                continue
        return best_result

