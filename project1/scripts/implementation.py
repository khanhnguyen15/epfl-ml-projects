import numpy as np

### least squares

def least_squares(y, tx):
    """
    apply least squares with normal linear basis
    returns: mse loss and optimized weights
    params: y (class labels)
            tx (features)
            lambda_ (regularization rate)
    """
    # num datapoints
    N = y.shape[0]
    
    # calculate the gram matrix
    gram = tx.T.dot(tx)
    
    # solving linear matrix equation gram * w = b 
    b = tx.T.dot(y)
    w = np.linalg.solve(gram, b)
    
    mse = 1 / (2 * N) * np.sum(np.square(y - tx.dot(w)))
    return mse, w


### ridge regression

def ridge_regression(y, tx, lambda_):
    """
    apply ridge regression with L2-norm
    returns: mse loss and optimized weights
    params: y (class labels)
            tx (features)
            lambda_ (regularization rate)
    """
    # num datapoints
    N = tx.shape[0]
    
    # num features
    d = tx.shape[1]
    
    # lambda_ap simpler notation: lampda_ap / (2 * N) = lambda
    lambda_ap = 2 * lambda_ * N
    i = np.identity(d)
    
    # solving linear matrix equation ax = b
    # in this case ((X^T) * X + lambda_ap * I) * w = (X^T) * y
    a = tx.T.dot(tx) + lambda_ap * i
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    mse = 1 / (2 * N) * np.sum(np.square(y - tx.dot(w)))
    return mse, w