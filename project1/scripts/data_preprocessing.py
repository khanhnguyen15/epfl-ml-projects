# -*- coding: utf-8 -*-
import numpy as np

# Cleaning the data - GJ
# def DataCleaning(tx):
#     '''
#     Set 990 values (Outliers) to 0 in the data set.
#     '''
#     cd = np.c_[np.ones(x.shape[0]), tx] # cd = clean data
    
#     for i in range(cd.shape[0]):
#         for j in range(cd.shape[1]):
#             # filter out outlier with 0
#             if cd[i, j] <= -990:
#                 cd[i, j] = 0
#     return cd

# data categorize
def categorize(y, tX, ids):
    '''divide the data set into 4 subgroups based on PRI_jet_num'''
    c_0 = np.where(tX[:, 22] == 0)[0]
    c_1 = np.where(tX[:, 22] == 1)[0]
    c_2 = np.where(tX[:, 22] == 2)[0]
    c_3 = np.where(tX[:, 22] == 3)[0]
    
    return (y[c_0], tX[c_0], ids[c_0]), (y[c_1], tX[c_1], ids[c_1]), (y[c_2], tX[c_2], ids[c_2]), (y[c_3], tX[c_3], ids[c_3])

def remove_col(tX):
    '''remove colum which are the same across the dataset (std = 0)'''
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

def change_y(y):
    y[y == -1.0] = 0
    return y

def remove_outliers(y, x, mean_x, std_x):
    '''remove any rows that contain outliers values'''
    not_outliers = np.array([True for _ in range(x.shape[0])])
    for i in range(x.shape[1]):
        not_outliers = not_outliers * (np.abs(x[:, i] - mean_x[i]) <= 3 * std_x[i])
    return y[not_outliers], x[not_outliers]