import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1.0 / 2.0 * np.mean(e ** 2)

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)

## task 1) least_squares_GD

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent"""
    # we initialize w to a zeros vector
    w = initial_w
    # Define parameters to store weight and loss
    loss = 0
    losses = []
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient, e = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # store w and loss
        losses.append(loss)
        # update w by gradient
        w = w - gamma * gradient
 
    return w, losses

### task 2) lest square SGD

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    
    """Returns a randomized minibatch for Stochastic Gradient Descent"""
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
    """Apply stochastic GD on input parameters, returns final loss and w"""
    w = initial_w
    N = y.shape[0]
    batch_size = 1
    num_batches = int(np.ceil(N / batch_size))
    
    for n_iter in range(max_iters):
        batch_losses = []
        #select a batch of size 1
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches):
            
            #compute gradient and update w according to rule
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            batch_loss = compute_loss(y, tx, w)
            batch_losses.append(batch_loss)
            
        loss = np.mean(batch_losses)
    return w, loss

### task 3) least squares

def least_squares(y, tx):
    """Apply least squares with normal linear matrix equations"""
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
    """Apply ridge regression with L2-norm"""
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
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def compute_sigmoid_loss(y, tx, w):
    """compute loss given by sigmoid function"""
    predictions = sigmoid(tx.dot(w))
    neg_losses_per_datapoint = -(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return neg_losses_per_datapoint.sum()

def compute_logistic_gradient(y, tx, w):
    """computes gradient given by update equation"""
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred - y) 
    return gradient

def learning_by_logistic_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_sigmoid_loss(y, tx, w) 
    gradient = compute_logistic_gradient(y, tx, w)
    w = w - gamma * gradient

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

def penalized_logistic_regression(y, tx, w, lambda_):
    """Return the loss and gradient."""
    loss = compute_sigmoid_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression"""
    # we initialize it to a zeros vector
    w = initial_w
    
    losses = []
    threshold = 0.1 # 1e-8

    # start the logistic regression
    for i in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)

        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    norm = sum(w ** 2)
    cost = w + lambda_ * norm / (2 * np.shape(w)[0])

    return w, cost