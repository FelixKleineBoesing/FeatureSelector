# Feature Selector

**Heavy Development! WIP!**

###Overview
This is package which handles an automatic feature selection with different algorithms and ensembles of them.
Feel free to contribute. 


###Installation
```
git clone https://github.com/FelixKleineBoesing/pyFeatSel.git && cd pyFeatSel
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

###Usage

This package implements different Feature-selecting Classes. 
They provide an minimalistic interface to Feature Selection. Nevertheless is fully customizable in adding different algorithms, evaluation function and thresholds for classification.
Besides, the package has a builtin k-fold cross validation. 

In minimum, you have to deliver your train-data and label, as well as the number of folds and objective to run the Feature Selector.
 ```
 train_data: pd.DataFrame()
 train_label: np.ndarray()
 comp_feat_selector = GreedySearch(train_data=train_data, train_label=train_label, k_folds=10,
                                          objective="classification")
 comp_feat_selector.run_selecting()
 ```
 

###Content

There are different algorithms that are used in this package and more to come.
Already implemented are now the following:

***Complete-Feature-Space-Search***

Inspect the whole feature space and chooses the best configuration. Guaranteed to find the global optimum.
Complexity is O(n)

***Greedy-Search***

Iteratively adds new features to the dataset by choosing those with the smallest error. Terminates if overall error 
haven´t decreased since X iterations. 

**Implemtation in progress**

***Tabu-Search***

Inspects the neighbordhood of a random initialized configuration and chooses the one with smallest error. Places the 
previous feature one the tabu list to avoid circling. Terminates if no better solution in neighborhood.
Might stuck around a local optimum if tabu list is too small. Complexity depends on the length of tabu list but in average 
lesser than Neighbordhood-Search. 


***SMAC-Search***

This Model uses SMACv3 (probs to the autoML girls and guys from Uni Freiburg) to find the optimal Selection of features.

***Genetic Algorithm***

lorem ipsum

***Ensemble***   

lorem ipsum