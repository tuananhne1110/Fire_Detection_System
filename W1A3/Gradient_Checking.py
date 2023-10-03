import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0, x)
    
    return s

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[: 20].reshape((5, 4))
    parameters["b1"] = theta[20: 25].reshape((5, 1))
    parameters["W2"] = theta[25: 40].reshape((3, 5))
    parameters["b2"] = theta[40: 43].reshape((3, 1))
    parameters["W3"] = theta[43: 46].reshape((1, 3))
    parameters["b3"] = theta[46: 47].reshape((1, 1))

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

import numpy as np

def gradient_check_n_test_case(): 
    np.random.seed(1)
    x = np.random.randn(4,3)
    y = np.array([1, 1, 0])
    W1 = np.random.randn(5,4) 
    b1 = np.random.randn(5,1) 
    W2 = np.random.randn(3,5) 
    b2 = np.random.randn(3,1) 
    W3 = np.random.randn(1,3) 
    b3 = np.random.randn(1,1) 
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    
    return x, y, parameters

import numpy as np
from dlai_tools.testing_utils import single_test, multiple_test

def forward_propagation_test(target):
    x, theta = 2, 4
    expected_output = 8
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target) 
        
def backward_propagation_test(target):
    x, theta = 3, 4
    expected_output = 3
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)

def gradient_check_test(target):
    x, theta = 3, 4
    expected_output = 7.814075313343006e-11
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)
    
    
def predict_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
     'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
     'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
     'b2': np.array([[  9.14954378e-05]])}
    expected_output = np.array([[True, False, True]])

    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)


# import numpy as np
# from testCases import *
# from public_tests import *
# from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector


def forward_propagation(x, theta):
    J = np.dot(theta, x)
    return J

x, theta = 2, 4
J = forward_propagation(x, theta)
print ("J = " + str(J))
forward_propagation_test(forward_propagation)


def backward_propagation(x, theta):
    dtheta = x
    return dtheta


x, theta = 3, 4
dtheta = backward_propagation(x, theta)
print ("dtheta = " + str(dtheta))
backward_propagation_test(backward_propagation)


def gradient_check(x, theta, epsilon=1e-7, print_msg=False):
    
    # Compute gradapprox using right side of formula (1). epsilon is small enough, you don't need to worry about the limit.
    thetaplus = theta + epsilon                               
    thetaminus = theta - epsilon                            
    J_plus = forward_propagation(x, thetaplus)                
    J_minus = forward_propagation(x, thetaminus)              
    gradapprox = (J_plus - J_minus) / (2 * epsilon) 
    
    # Check if gradapprox is close enough to the output of backward_propagation()
    grad = backward_propagation(x, theta)
    
    numerator = np.linalg.norm(grad - gradapprox)                      # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)    # Step 2'
    difference = numerator / denominator
    
    if print_msg:
        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference


x, theta = 3, 4
difference = gradient_check(x, theta, print_msg=True)



def forward_propagation_n(X, Y, parameters):
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    log_probs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(log_probs)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache



def backward_propagation_n(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) * 2
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        thetaplus =  np.copy(parameters_values)                                       
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                   
        J_plus[i], _ =  forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))  

        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)                                       
        thetaminus[i][0] = thetaminus[i][0] - epsilon                                       
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus)) 
        
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)                                     
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                   
    difference = numerator / denominator                                              
    
    if print_msg:
        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference


X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y, 1e-7, True)
expected_values = [0.2850931567761623, 1.1890913024229996e-07]
assert not(type(difference) == np.ndarray), "You are not using np.linalg.norm for numerator or denominator"
assert np.any(np.isclose(difference, expected_values)), "Wrong value. It is not one of the expected values"





