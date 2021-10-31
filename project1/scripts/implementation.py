import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)

## task 1) least_squares_GD

def least_squares_GD(y, tx,initial_w,max_iters, gamma):
    """ Linear regression using gradient descent
    """
    # we initialize w to a zeros vector
    w = initial_w
    # Define parameters to store weight and loss
    loss = 0
    losses = []
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient,err = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # store w and loss
        losses.append(loss)
        # update w by gradient
        w = w - gamma * gradient
 
    return w, losses
### task 2) lest square SGD

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    
    """returns a randomized minibatch for Stochastic Gradient Descent"""
    data_size = len(y)

    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = tx[shuffle_indices]
    
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """does stochastic GD on input parameters, returns final loss and w"""
    w = initial_w
    for n_iter in range(max_iters):
        
        #select a batch of size 1
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            
            #compute gradient and update w according to rule
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)
            
    return w, loss

### task 3) least squares

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
    
    loss = compute_loss(y, tx, w)
    return w, loss


### task 4) ridge regression

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
    
    loss = compute_loss(y, tx, w)
    return w, loss

## task 5) logistic regression

def sigmoid(t):
    """compute sigmoid function"""
    expo = np.exp(-t)
    result = 1.0/(1.0 + expo)
    return result

def compute_sigmoid_loss(tx, y, w):
    """compute loss given by sigmoid function"""
    predictions = sigmoid(tx @ w)
    neg_losses_per_datapoint = -(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return neg_losses_per_datapoint.sum()

def compute_logistic_gradient(tx, y, w):
    """computes gradient given by update equation"""
    pred = sigmoid(tx @ w)
    gradient = tx.T @ (pred - y) 
    return gradient


def learning_by_logistic_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    # ***************************************************
    loss = compute_sigmoid_loss(tx, y, w) 
    # ***************************************************
    gradient = compute_logistic_gradient(tx, y, w)
    # ***************************************************
    w = w - gamma * gradient
    # ***************************************************
    return loss, w 

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_logistic_gradient_descent(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            return w, loss
    return w, loss

## task 6) reg_logistic_regression

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad
def calculate_loglikelihood_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    loss = calculate_loglikelihood_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
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

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression"""
   # we initialize it to a zeros vector
    w = initial_w
    
    losses = []
    threshold = 0.1 # 1e-8

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