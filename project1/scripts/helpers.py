import numpy as np
from implementation import ridge_regression

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred

def accuracy_score(y_pred, y_label):
    return (y_pred == y_label).sum() / y_label.shape[0]

def f1_score(y_pred, y_label):
    return 

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    N = x.shape[0]
    d = x.shape[1]
    feature_matrix = np.zeros((N, d * degree))
    for i in range(d):
        for j in range(1, degree + 1):
            feature_matrix[:, i * degree + j - 1] = np.power(x[:, i], j)
    return feature_matrix

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def ridge_kth_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    
    ind_test = k_indices[k]
    ind_train = np.delete(k_indices, k, 0).flatten()
    
    y_test = y[ind_test]
    x_test = x[ind_test]
    
    y_train = y[ind_train]
    x_train = x[ind_train]
    
    _, weights = ridge_regression(y_train, x_train, lambda_)
    y_pred = predict_labels(weights, x_train)
    
    return accuracy_score(y_pred, y_train)

def ridge_cross_validation(y, x, lambdas):
    seed = 42
    k_fold = 5
    
    k_indices = build_k_indices(y, k_fold, seed)
    
    accuracies = []
    for lambda_ in lambdas:
        fold_accuracies = []
        for k in range(k_fold):
            kth_accuracy = ridge_kth_validation(y, x, k_indices, k, lambda_)
            fold_accuracies.append(kth_accuracy)
        accuracies.append(np.mean(fold_accuracies))
    
    return accuracies

def poly_cross_validation(y, x, best_lambda, polys):
    seed = 42
    k_fold = 5
    
    k_indices = build_k_indices(y, k_fold, seed)
    
    accuracies = []
    for degree in polys:
        fold_accuracies = []
        x_temp = build_poly(x, degree)
        for k in range(k_fold):
            kth_accuracy = ridge_kth_validation(y, x_temp, k_indices, k, best_lambda)
            fold_accuracies.append(kth_accuracy)
        accuracies.append(np.mean(fold_accuracies))
    
    return accuracies