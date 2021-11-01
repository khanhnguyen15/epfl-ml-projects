# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt


# importing other functions
from data_preprocessing import *
from implementation import *
from helpers import *
from plot import *

# categorizing data

print("Loading train data ...")
from proj1_helpers import *
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

print("Categorizing train data ...")
groups = categorize(y, tX, ids)

# removing missing values columns
for i, (y, tX, ids) in enumerate(groups):
    tX = remove_uniform_col(tX)
    groups[i] = (y, tX, ids)

# storing non_correlated_cols to apply to testing set
non_correlated_cols = []

# cleaning and preprocessing the data
print("Cleaning train data ...")
for i, (y, tX, ids) in enumerate(groups):
    y = change_y(y)
    non_correlated_cols.append(non_correlated_col(tX))
    
    x, mean_x, std_x = clean(tX)
    y, x = remove_outliers(y, x, mean_x, std_x)
    x = x[:, non_correlated_cols[i]]
    
    groups[i] = (y, x, ids)
    
# add polynomial basis of 6
print("Building polynomial for train data ...")
for i, (y, x, ids) in enumerate(groups):
    x = build_poly(x, 6)
    groups[i] = (y, x, ids)
    
# using least_squares to get the weights
print("Applying least squares ...")
weights = []
y_preds = []

for y, x, ids in groups:
    w, _ = least_squares(y, x)
    y_pred = predict_labels(w, x)
    
    weights.append(w)
    y_preds.append(y_pred)

y_pred = np.concatenate(y_preds)
y_label = np.concatenate([y for (y, _, _) in groups])

accuracy_score(y_pred, y_label)

print("MACHINE LEARNING DONE!")

# getting test data set and do the similar steps of categorizing and preprocessing
print("Loading test data ...")
DATA_TEST_PATH = '../data/test.csv'
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

print("Categorizing test data ...")
groups_test = categorize(y_test, tX_test, ids_test)

for i, (y_test, tX_test, ids_test) in enumerate(groups_test):
    tX_test = remove_uniform_col(tX_test)
    groups_test[i] = (y_test, tX_test, ids_test)
    
    
print("Cleaning test data ...")
for i, (y_test, tX_test, ids_test) in enumerate(groups_test):
    y_test = change_y(y_test)
    
    x_test, mean_x_test, std_x_test = clean(tX_test)
    x_test = x_test[:, non_correlated_cols[i]]
    
    groups_test[i] = (y_test, x_test, ids_test)
    
# building polynomial
print("Building polynomial for test data ...")
for i, (y_test, x_test, ids_test) in enumerate(groups_test):
    x_test = build_poly(x_test, 6)
    groups_test[i] = (y_test, x_test, ids_test)
    
# generating predictions
print("Genrating predictions ...")
y_preds_test = []
for i, (y_test, x_test, ids_test) in enumerate(groups_test):
    y_pred_test = predict_labels(weights[i], x_test)
    y_preds_test.append(y_pred_test)
    
y_pred_test = np.concatenate(y_preds_test)
y_pred_test = unchange_y(y_pred_test)

g_ids_test = np.concatenate([ids_test for (_, _, ids_test) in groups_test])

# saving the submission file
print("Saving submission ...")
OUTPUT_PATH = '../data/sample-submission.csv'
create_csv_submission(g_ids_test, y_pred_test, OUTPUT_PATH)

print("DONE!")