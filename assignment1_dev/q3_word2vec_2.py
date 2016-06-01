import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N,1)) + 1e-30
    
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (np.amax(np.fabs(x - np.array([[0.6,0.8],[0.4472136,0.89442719]]))) <= 1e-6)
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  

    # print '---------'
    # print 'target word index: '
    # print target

    # print 'predicted word: '
    # print predicted
    # print 'predicted.shape: ', predicted.shape
    # ### YOUR CODE HERE
    # print 'outputVectors.T.shape: ', outputVectors.T.shape
    # score = predicted.dot(outputVectors.T)
    # print 'score: ', score
    # probabilities = softmax(score)
    # print 'probablities: '
    # print probabilities
    # cost = -np.log(probabilities[target])
    # delta = probabilities
    # delta[target] -= 1
    # print 'outputVectors:'
    # print outputVectors
    # print 'delta: '
    # print delta
    # N = delta.shape[0]
    # D = predicted.shape[0]
    # print 'delta.reshape((N,1)): ', delta.reshape((N,1))
    # grad = delta.reshape((N,1)) * predicted.reshape((1,D))
    # print 'grad: ', grad
    # gradPred = (delta.reshape((1,N)).dot(outputVectors)).flatten()
    # print 'gradPred: ', gradPred
    


    score = predicted.dot(outputVectors.T)
    probabilities = softmax(score)
    cost = -np.log(probabilities[target])
    delta = probabilities
    delta[target] -= 1
    N = delta.shape[0]
    D = predicted.shape[0]
    grad = delta.reshape((N,1)) * predicted.reshape((1,D))
    gradPred = (delta.reshape((1,N)).dot(outputVectors)).flatten()    
    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    #negative_samples = [dataset.sampleTokenIdx() for i in range(K)]

    v_c = predicted
    u_o = outputVectors[target]
    U = outputVectors

    grad = np.zeros(U.shape)
    gradPred = np.zeros(v_c.shape)
    negative_samples = []
    for i in xrange(K):
        newidx = dataset.sampleTokenIdx() # why is our outputVectors our entire dataset??? Shouldn't outputVectors be window, and dataset our entire vocabulary?
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        negative_samples.append(newidx)

    
    negative_vectors = outputVectors[negative_samples]

    f1 = np.dot(u_o.T, v_c)
    print 'f1: ', f1
    sig1 = sigmoid(f1)


    f2 = negative_vectors.dot(v_c)
    print 'f2: ', f2
    sig2 = sigmoid(-f2)

    c1 = -np.log(sig1)
    c2 = -np.sum(np.log(sig2))
    cost = c1 + c2 

    print 'c1: ', c1
    print 'c2: ', c2
    print 'cost: ', cost





## THE SOLUTION CODE PROVIDE
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    
    indices = [target]
    indices += negative_samples

    # for k in xrange(K):
    #     newidx = dataset.sampleTokenIdx()
    #     while newidx == target:
    #         newidx = dataset.sampleTokenIdx()
    #     indices += [newidx]
        
    labels = np.array([1] + [-1 for k in xrange(K)])
    vecs = outputVectors[indices,:]
    f1_fasit =  vecs.dot(predicted)
    print 'f1_fasit: ', f1_fasit
    t = sigmoid(vecs.dot(predicted) * labels)
    cost = -np.sum(np.log(t))
    print 'solution cost: ', cost
    

    # check the derivative of J wrt v_c in the solution set
    first = sig1 - 1
    second = sig2 - 1
    first_multiplied_with_u = np.dot(first, u_o)
    second_multiplied_with_u = np.dot(second, negative_vectors)
    gradPred = first_multiplied_with_u - second_multiplied_with_u
    print 'gradPred: ', gradPred
    print 'first: ', first

    delta = labels * (t - 1)
    print 'delta: '
    print delta
    print 'vecs: ' 
    print vecs

    #gradPred = delta.reshape((1,K+1)).dot(vecs).flatten()
    print 'gradPred: ', gradPred
    # gradtemp = delta.reshape((K+1,1)).dot(predicted.reshape(
    #     (1,predicted.shape[0])))
    # for k in xrange(K+1):
    #     grad[indices[k]] += gradtemp[k,:]


    print 'second: '
    print second
    print 'v_c'
    print v_c

    grad_first = first * v_c
    grad_second = second * v_c

    ## THIS WAS GREYED OUT FROM THE BEGINNING
#     t = sigmoid(predicted.dot(outputVectors[target,:]))
#     cost = -np.log(t)
#     delta = t - 1

#     gradPred += delta * outputVectors[target, :]
#     grad[target, :] += delta * predicted
    
#     for k in xrange(K):
#         idx = dataset.sampleTokenIdx()
        
#         t = sigmoid(-predicted.dot(outputVectors[idx,:]))
#         cost += -np.log(t)
#         delta = 1 - t

#         gradPred += delta * outputVectors[idx, :]
#         grad[idx, :] += delta * predicted
    
    ### END YOUR CODE
    return costt, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    
    currentI = tokens[currentWord]
    predicted = inputVectors[currentI, :]
    
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for cwd in contextWords:
        idx = tokens[cwd]
        cc, gp, gg = word2vecCostAndGradient(predicted, idx, outputVectors, dataset)
        cost += cc
        gradOut += gg
        gradIn[currentI, :] += gp
    
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    
    # D = inputVectors.shape[1]
    # predicted = np.zeros((D,))
    
    # indices = [tokens[cwd] for cwd in contextWords]
    # for idx in indices:
    #     predicted += inputVectors[idx, :]
    
    # cost, gp, gradOut = word2vecCostAndGradient(predicted, tokens[currentWord], outputVectors, dataset)
    # gradIn = np.zeros(inputVectors.shape)
    # for idx in indices:
    #     gradIn[idx, :] += gp
    
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()