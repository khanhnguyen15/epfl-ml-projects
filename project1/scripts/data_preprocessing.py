# -*- coding: utf-8 -*-
import numpy as np

# data categorization
def categorize(y, tX, ids):
    """Divide the data set into 4 subgroups based on PRI_jet_num"""
    c_0 = np.where(tX[:, 22] == 0)[0]
    c_1 = np.where(tX[:, 22] == 1)[0]
    c_2 = np.where(tX[:, 22] == 2)[0]
    c_3 = np.where(tX[:, 22] == 3)[0]
    
    return [(y[c_0], tX[c_0], ids[c_0]), (y[c_1], tX[c_1], ids[c_1]), (y[c_2], tX[c_2], ids[c_2]), (y[c_3], tX[c_3], ids[c_3])]

def remove_uniform_col(tX):
    """Remove colum which are the same across the dataset (std = 0)"""
    col_to_remove = [22]
    for col in range(tX.shape[1]):
        if np.std(tX[:, col]) == 0:
            col_to_remove.append(col)
    col_to_keep = [i for i in range(tX.shape[1]) if i not in col_to_remove]
    return tX[:, col_to_keep]

# data cleaning
def clean(X):
    x = np.copy(X)
    """Remove weird values from the original data set."""
    x[abs(x) ==  999] = np.nan
    mean_x = np.nanmean(x, axis=0)
    std_x = np.nanstd(x, axis=0)
    rows, cols = x.shape
    for i in range(rows):
        for j in range(cols):
            if(np.isnan(x[i][j])):
                x[i][j] = mean_x[j]
                
    return x, mean_x, std_x

def standardize(x, mean_x, std_x):
    """Standardize values from the cleaned data set."""
    tX = np.copy(x)
    tX = tX - mean_x[np.newaxis, :]
    tX = tX / std_x[np.newaxis, :]
    return tX

def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x

def change_y(y):
    """Change label from [-1, 1] to [0, 1]"""
    y[y == -1.0] = 0
    return y

def unchange_y(y):
    """Change label from [0, 1] to [-1, 1]"""
    y[y == 0] = -1
    return y

def non_correlated_col(tX, threshold=0.9):
    """
    Eliminate the columns which are highly correlated to another one in the features matrix.
    Default threshold is 0.9.
    Return the list of columns that we keep (needed to apply to the test set)
    """
    col_to_remove = []
    corr_matrix = np.corrcoef(tX, rowvar=False)
    for i in range(tX.shape[1]):
        for j in range(i):
            if (corr_matrix[i, j] > threshold) and (j not in col_to_remove):
                col_to_remove.append(i)
    col_to_keep = [i for i in range(tX.shape[1]) if i not in col_to_remove]
    return col_to_keep

def remove_outliers(y, x, mean_x, std_x):
    """Remove any rows that contain outliers values"""
    not_outliers = np.array([True for _ in range(x.shape[0])])
    for i in range(x.shape[1]):
        not_outliers = not_outliers * (np.abs(x[:, i] - mean_x[i]) <= 3 * std_x[i])
    return y[not_outliers], x[not_outliers]