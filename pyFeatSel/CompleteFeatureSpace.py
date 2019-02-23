from pyFeatSel.FeatureSelector import FeatureSelector
from pyFeatSel.Helper import create_k_fold_indices, create_all_combinations



class CompleteFeatureSpace(FeatureSelector):

    def run_selecting(self):
        column_names = self.train_data.columns.values.tolist()

        combns = [comb for comb in create_all_combinations(column_names) if len(comb) > 0]

        for comb in combns:
            measure = self.inner_run(list(comb))
            self.computed_features += [{"measure": measure, "column_names": comb}]
            if measure["test"] < self.best_result["measure"]["test"] or self.best_result is None:
                self.best_result = {"measure": measure, "column_names": comb}

        print(1)


