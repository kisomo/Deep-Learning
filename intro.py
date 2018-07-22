import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
#import grading
#grader = grading.Grader(assignment_key="UaHtvpEFEee0XQ6wjK-hZg", 
#                      all_parts=["xU7U4", "HyTF6", "uNidL", "ToK7N", "GBdgZ", "dLdHG"])

# token expires every 30 min
COURSERA_TOKEN = 1 ### YOUR TOKEN HERE
COURSERA_EMAIL = 1 ### YOUR EMAIL HERE

with open('/home/terrence/CODING/Python/MODELS/intro-to-dl/week1/train.npy', 'rb') as fin:
    X = np.load(fin)
    
with open('/home/terrence/CODING/Python/MODELS/intro-to-dl/week1/target.npy', 'rb') as fin:
    y = np.load(fin)

#print(X[:3,:])
print(X.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
plt.show()

def expand(X):
    """
    Adds quadratic features. 
    This expansion allows your linear model to make non-linear separation.
    
    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
    
    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    X_expanded = np.zeros((X.shape[0], 6))
    
    # TODO:<your code here>
    X_expanded[:,0] = X[:,0]
    X_expanded[:,1] = X[:,1]
    X_expanded[:,2] = X[:,0]**2
    X_expanded[:,3] = X[:,1]**2
    X_expanded[:,4] = X[:,0]*X[:,1]
    X_expanded[:,5] = 1

    return X_expanded


X1 = expand(X)
#print(X1[:3,:])

#tests
assert isinstance(X1,np.ndarray)#"please make sure you return numpy array"
print(X1.shape)
print("Seems legit!")
print(y[:10])


def probability(X, w):
    """
    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x), see description above
        
    Don't forget to use expand(X) function (where necessary) in this and subsequent functions.
    
    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """

    # TODO:<your code here>
    a = expand(X)
    b = np.dot(a,w)
    c = 1/(1+np.exp(-b))
    return c

w = np.linspace(-1,1,6)
res = probability(X,w)
print(res.shape)
print(res[:10])

def compute_loss(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute scalar loss function using formula above.
    """
    # TODO:<your code here>
    a = - np.sum(np.multiply(y, np.log(probability(X,w))) + np.multiply((1-y), np.log(1-probability(X,w))))
    return a

res2 = compute_loss(X,y,w)
print(res2)

## GRADED PART, DO NOT CHANGE!
#grader.set_answer("HyTF6", ans_part2)
## you can make submission with answers so far to check yourself at this stage
#grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)

def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each weights.
    """
    
    # TODO<your code here>

    