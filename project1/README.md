# EPFL - Machine Learning Project 1 

**Team:** CGK

**Team Members:** Cesar Descalzo, Gabriel Juri, Cao Khanh Nguyen

In this project, the task is to utilize classification techniques to predict whether a collision event can produce Higgs boson particles, given the measured "decay signature". In our work, we investigated a variety of classification techniques and experimented different approach to preprocessing the data. Finally, we introduce a pipeline that can accurately predict 81.8% of the testing datasets with an F1-score of 0.727, using least squares method with linear equation.

**Instructions**:
1. Download zip data file from [AiCrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs). Unzip the file and put `train.csv` and `test.csv` into the './data/' folder.
3. `cd` into the './scripts/' folder and run `run.py`

## Notebook

There are three notebooks inside the projects. All of them are utilized duringn the experimental process of the project.
- **`pandas_insights`**: Use `pandas` to examine the dataframe. The use of `pandas` is only for EDA purpose and is not included in the scope of the project.
- **`project1_non_normalized`**: Apply different machine learning methods to the processed dataset without normalization.
- **`project1_normalized`**: Apply different machine learning methods to the processed dataset with normalization.

## External modules

We also included our own implemented modules to help with the challenge

- **`proj1_helpers.py`**: Provided helpers functions for loading and creating submission. We commnted out the `predict_labels` to use our own methods. Otherwise the file is left untouched.
- **`data_preprocessing.py`**: Module to preprocess the data. Preprocessing functions include categorization, missing values correction, correlation detection, outliers removal, relabeling, and standardizing.
- **`helpers.py`**: Helpers module to handle model building and selection. Functions include create prediction labels, assess accuracy, and cross validations.
- **`plot.py`**: Module to plot graphs that would be supplied to the report.
- **`run.py`**: Runnable script that can be invoke to generate a final submission file to submit to [AiCrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) challenge. The default location for saving submission is the '../data/' folder

Further information regarding each function can be found in the function description inside each module.
