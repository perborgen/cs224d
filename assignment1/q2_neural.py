import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    #z1 = data.dot(W1) + b1
    #hidden = sigmoid(z1)
    #z2 = hidden.dot(W2) + b2
    #print 'z2.shape: ', z2.shape
    #prediction = softmax(z2)
    ### END YOUR CODE
    
    hidden = sigmoid(data.dot(W1) + b1)
    prediction = softmax(hidden.dot(W2) + b2)
    cost = -np.sum(np.log(prediction) * labels)

    
    ### YOUR CODE HERE: backward propagation
    #print 'NN: ', Dx, H, Dy
    #print 'b1.shape: ', b1.shape
    #print 'prediction.shape: ', prediction.shape
    #print 'labels.shape : ', labels.shape
    #print 'W2.shape: ', W2.shape
    #print 'hidden.shape: ', hidden.shape
    #print 'hidden.T.shape: ', hidden.T.shape
    #print 'delta.shape: ', delta.shape
    #print 'W1.shape: ', W1.shape
    #print 'data.shape: ', data.shape
    #gradW2 = delta * hidden
    #print 'sigmoid_grad(hidden).shape: ', sigmoid_grad(hidden).shape
    delta = prediction - labels
    gradW2 = hidden.T.dot(delta)
    gradb2 = np.sum(delta, axis = 0)
    hidden_delta = delta.dot(W2.T) * sigmoid_grad(hidden)
    gradW1 = data.T.dot(hidden_delta)
    gradb1 = np.sum(hidden_delta, axis = 0)
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()