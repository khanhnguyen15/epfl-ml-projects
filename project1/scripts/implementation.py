import numpy as np
## task 1) least_squares_GD

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent
    """
    # we initialize it to a zeros vector
    initial_w = np.zeros(tx.shape[1])

    # Define parameters to store weight and loss
    loss = 0
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w = gamma * gradient
 
    return w, loss

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



## task 6) reg_logistic_regression

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return loss, w

def reg_logistic_regression(y, tx, lambda_, max_iters, gamma):
    """Regularized logistic regression"""
   # we initialize it to a zeros vector
        initial_w = np.zeros(tx.shape[1])

    w = initial_w
    
    losses = []
    threshold = 0.1 # 1e-18

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)

        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    norm = sum(w ** 2)
    cost = w + lambda_ * norm / (2 * np.shape(w)[0])

    return w, cost
