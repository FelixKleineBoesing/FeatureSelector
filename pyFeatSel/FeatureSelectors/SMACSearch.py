import logging
import numpy as np
import pandas as pd
import types
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    NumericalHyperparameter, IntegerHyperparameter, FloatHyperparameter, \
    NormalIntegerHyperparameter, NormalFloatHyperparameter, OrdinalHyperparameter, \
    Constant

from smac.facade.smac_facade import SMAC
from smac.runhistory.runhistory import RunKey
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncArray
from pyFeatSel.Evaluator.Evaluator import EvaluatorBase, RMSE, Accuracy
from pyFeatSel.Models.Model import Model
from pyFeatSel.misc.Helpers import create_k_fold_indices, threshold_base
from pyFeatSel.Models.Model import XGBoostModel
from pyFeatSel.FeatureSelectors.FeatureSelector import FeatureSelector


class SMACSearch(FeatureSelector):

    def __init__(self, train_data: pd.DataFrame, train_label: np.ndarray,objective: str = "classification",
                 k_folds: int = 1, evaluator: EvaluatorBase = None, model: Model = None,
                 maximize_measure: bool = None, threshold_func: types.FunctionType = None,
                 maxtime: int = None, maxrun: int = None):
        if maxtime is None and maxrun is None:
            logging.info("Runtime will be default 60 seconds!")
            self.maxtime = 60
        elif maxtime is not None:
            self.maxtime = maxtime
            self.maxrun = None
        else:
            self.maxrun = maxrun
            self.maxtime = None
        super().__init__(train_data, train_label, objective, k_folds, evaluator, model,
                         maximize_measure, threshold_func)

    def run_selecting(self):
        # TODO coonfigpsace seems not to work
        column_names = self.train_data.columns.values.tolist()

        scenario_dict = {"run_obj": "quality",
                         "deterministic": "true",
                         "initial_incumbent": "DEFAULT",
                         "output_dir": None}
        if self.maxtime is not None:
            scenario_dict["wallclock_limit"] = self.maxtime
        else:
            scenario_dict["runcount_limit"] = self.maxrun
        cs = ConfigurationSpace()
        for column_name in column_names:
            param = UniformIntegerHyperparameter(name=column_name, lower=float(0), upper=float(1),
                                                 q=None, log=False, default_value=float(1))
            cs.add_hyperparameter(param)

        scenario = Scenario(scenario_dict)
        scenario.cs = cs

        ta = ExecuteTAFuncArray(ta=self.run_pipeline, use_pynisher=False)

        smac = SMAC(scenario=scenario, tae_runner=ta, rng=1)
        #smac.logger = logging.getLogger(smac.__module__ + "." + smac.__class__.__name__)
        incumbent, runhistory = smac.optimize()

        config_id = smac.solver.runhistory.config_ids[incumbent]
        run_key = RunKey(config_id, None, 0)
        incumbent_performance = smac.solver.runhistory.data[run_key]

        incumbent = np.array([incumbent[idx]
                              for idx in incumbent.keys()], dtype=np.float)

    def run_pipeline(self, args):
        column_names = [key for key in args._values.keys() if args._values[key] == 1]
        print(column_names)
        measure = self.inner_run(column_names)
        self.computed_features += [{"measure": measure, "column_names": column_names}]
        logging.info("Test Measure: {0}, Val Measure: {1}".format(measure["test"], measure["val"]))
        if self.best_result is None:
            self.best_result = {"measure": measure, "column_names": column_names}
        elif measure["test"] > self.best_result["measure"]["test"] and self.maximize_measure:
            self.best_result = {"measure": measure, "column_names": column_names}
        elif measure["test"] < self.best_result["measure"]["test"] and not self.maximize_measure:
            self.best_result = {"measure": measure, "column_names": column_names}
        if self.maximize_measure:
            # invert return value because smac is only able to minimize
            return 1/measure["test"]
        else:
            return measure["test"]
