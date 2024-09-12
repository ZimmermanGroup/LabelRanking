# LabelRanking
 
### Contents
1. label_ranking.py

   Implementation of various label ranking algorithms to be compatible with sklearn functionalities like GridSearchCV.
* rank_aggregation.py :
Includes borda and soft rank aggregation methods used for label ranking algorithms.

2. evaluate_benchmark.py

   Script to evaluate the implemented label ranking algorithms on a subset of datasets that are commonly considered in the label ranking literature. 
* If you want to evaluate the algorithms rpc and ibm on the datasets housing and iris on the dataset with 30% datapoints randomly marked as missing and save the results in an excel file, run the command
`python evaluate_benchmark.py --algorithms rpc --algorithms ibm --datasets housing --datasets iris --missing_portion 0.3 -s`

3. dataloader.py

   Prepares dataset objects that are considered in the study.  
   Onehot, fingerprint or descriptor (if available) input arrays and ranking, yield, output arrays etc. can be accessed as attributes.
* datasets folder:
includes the raw datasets.
* dataset_structure_analysis.ipynb:
analyzing the portion of each reaction condition being the top-performant, plots that are in the first half of the supporting info.

4. evaluator.py

   Prepares evaluator objects for different algorithms.  
   By specifying how the evaluations should be done (such as which features to use, cross validation, number of reaction conditions to sample) and feeding the dataset object from above, produces a dictionary that records the performances.

5. executor.py

   Script for executing the evaluations.
* If you want to evaluate the baseline (of uniformly choosing highest average yielding condition), rpc, random forest regressor on the NatureHTE amine dataset featurized as fingerprints with one missing reaction condition for every substrate in the training dataset, run the command
`python executor.py --dataset NatureHTE --feature fp --label_component amine --baseline --rpc --rfr --n_missing_reaction 1`

6. performance_analysis.ipynb

   Conducts various analyses on the performance achieved by different algorithms with different datasets with the excel files saved by executor.py.

7. plotting_utils.py

   Includes various functions that draw the plots in the jupyter notebooks.