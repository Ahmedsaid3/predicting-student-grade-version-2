# predicting-student-grade-dev
## Figures
**Figure 2 (elbow method):** experiments/cluster_model_collaborative_filtering_Student_based with KMeans.ipynb <br /> <br />
**Table 2 and 3 (hyperparameters):** hyperparameters folder <br /> <br />
**Figure 3:** preprocess folder <br /> <br />
**Figure 4 and 5:** experiments/non-time series analysis.ipynb <br /> <br /> <br />

**Figure 6 and Figure 7 (Baseline):** <br /> -- prediction of values, experiment and production of result json file on experiments/baseline_collaborative_filtering.ipynb and experiments/baseline_regression.ipynb <br /> 
-- visualization and production of graph on visualization/comparisons.ipynb <br /> <br /> <br />
**Figure 8 and Figure 9 (Baseline with subset):** <br /> -- prediction of values, experiment and production of result json file on experiments/baseline_collaborative_filtering.ipynb and experiments/baseline_regression.ipynb <br />
-- visualization and production of graph on visualization/comparisons.ipynb <br /> <br /> <br />
**Figure 10 and Figure 11 (Student based):** <br /> -- prediction of values, experiment and production of result json file on experiments/cluster_model_collaborative_filtering_Student_based with KMeans.ipynb <br />
-- visualization and production of graph on visualization/comparisons.ipynb <br /> <br /> <br />
**Figure 12 and Figure 13 (Course based):** <br /> -- prediction of values, experiment and production of result json file on experiments/cluster_model_collaborative_filtering_Course_based with KMeans.ipynb <br />
-- visualization and production of graph on visualization/comparisons.ipynb <br /> <br /> <br />
                         
**Figure 14 and Figure 15 (SOTA):** at the bottom of visualization/comparisons.ipynb file <br /> <br /> <br />


// Won't be used anymore <br />
**Figure 16 and 17 (Weighted SOTA):** Since we no longer will use weighted average we remowed them but they are in visualizations/chinese_dataset_visualizations/comparisons_student_based.ipynb in older version <br /> <br />


## Notes
-- Mimis et al. Naive Bayes and Neural Network experiments are done on weka. Experiments with subset are not done yet on weka. Naive Bayes and Neural Network will be run on subset on weka. <br />
-- Getting no fallback results for cf-ibrahimzada and cb-cakmak again may be necessary, after algorithm change the subset mask is also change. The relevant scripts are at the end of the relevant experiment files (experiments/cluster_model__collaborative_filtering__Student based with KMeans.ipynb and experiments/2017 paper.ipynb). <br />
-- Since ANHUI dataset has a brand new algorithm, all of the experiments are out-dated hence we removed them. But they are still present in older version. <br />
