# -*- coding: utf-8 -*-

# Cleaning the data - GJ
def DataCleaning(tx):
    '''
    Set 990 values (Outliers) to 0 in the data set.
    '''
    cd = np.c_[np.ones(x.shape[0]), tx] # cd = clean data
    
    for i in range(cd.shape[0]):
        for j in range(cd.shape[1]):
            # filter out outlier with 0
            if cd[i, j] <= -990:
                cd[i, j] = 0
    return cd
