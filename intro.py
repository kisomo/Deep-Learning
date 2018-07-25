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
    #Adds quadratic features. 
    #This expansion allows your linear model to make non-linear separation.
    #For each sample (row in matrix), compute an expanded row:
    #[feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
    #:param X: matrix of features, shape [n_samples,2]
    #:returns: expanded features of shape [n_samples,6]
    
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
    #Given input features and weights
    #return predicted probabilities of y==1 given x, P(y=1|x), see description above
        
    #Don't forget to use expand(X) function (where necessary) in this and subsequent functions.
    
    #:param X: feature matrix X of shape [n_samples,6] (expanded)
    #:param w: weight vector w of shape [6] for each of the expanded features
    #:returns: an array of predicted probabilities in [0,1] interval.
    

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
    #Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    #and weight vector w [6], compute scalar loss function using formula above.
    
    # TODO:<your code here>
    a = - np.sum(np.multiply(y, np.log1p(probability(X,w))) + np.multiply((1-y), np.log1p(1-probability(X,w))))
    return a

res2 = compute_loss(X1,y,w)
print(res2)

## GRADED PART, DO NOT CHANGE!
#grader.set_answer("HyTF6", ans_part2)
## you can make submission with answers so far to check yourself at this stage
#grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)

def compute_grad(X, y, w):
    #Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    #and weight vector w [6], compute vector [6] of derivatives of L over each weights.    
    # TODO<your code here>
    a = y - probability(X,w)
    res = np.dot(a,X)
    return res
    
res3 = compute_grad(X1,y,w)
print(res3)

# use output of this cell to fill answer field 
ans_part3 = np.linalg.norm(compute_grad(X1, y,w))
print(ans_part3)

## GRADED PART, DO NOT CHANGE!
#grader.set_answer("uNidL", ans_part3)

## you can make submission with answers so far to check yourself at this stage
#grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)

from IPython import display

h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def visualize(X, y, w, history):
    #draws classifier prediction with matplotlib magic
    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    display.clear_output(wait=True)
    plt.show()

visualize(X1,y,w,[0.5, 0.5, 0.25])

# ---------------------------------------------------- SGD ----------------------------------------------------------
print("------------------------SGD ---------------------")
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])
print(w)
eta= 0.1 # learning rate

n_iter = 100
batch_size = 1
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X1.shape[0], batch_size)
    loss[i] = compute_loss(X1, y, w)
    #if i % 10 == 0:
    #    visualize(X1[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    w = w - eta*compute_grad(X1[ind,:],y[ind],w)

visualize(X1, y, w, loss)
plt.clf()

print(w)
#print("---------------------------")
#print(loss)

# ---------------------------------------------------- mini-batch ----------------------------------------------------------

# please use np.random.seed(42), eta=0.1, n_iter=100 and batch_size=4 for deterministic results
print("-----------------------mini-batch ----------------------------")
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])
print(w)
eta= 0.1 # learning rate

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X1.shape[0], batch_size)
    loss[i] = compute_loss(X1, y, w)
    #if i % 10 == 0:
    #    visualize(X1[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    w = w - eta*compute_grad(X1[ind,:],y[ind],w)

visualize(X1, y, w, loss)
plt.clf()

print(w)
#print("---------------------------")
#print(loss)

# ----------------------------------------------------mini-batch with momentum -----------------------------------------------------
print("-----------------------------mini-batch with momentum -----------------------------------------")
# please use np.random.seed(42), eta=0.05, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])
print(w)
eta = 0.05 # learning rate
alpha = 0.9 # momentum
nu = np.zeros_like(w)

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X1.shape[0], batch_size)
    loss[i] = compute_loss(X1, y, w)
    #if i % 10 == 0:
    #    visualize(X1[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    nu = alpha*nu + eta*compute_grad(X1[ind,:],y[ind],w)
    w = w - nu

visualize(X1, y, w, loss)
plt.clf()

print(w)


# ----------------------------------------------------Nesterov momentum -----------------------------------------------------
print("--------------------------Nesterov momentum -----------------------------------------")
# please use np.random.seed(42), eta=0.05, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])
print(w)
eta = 0.05 # learning rate
alpha = 0.9 # momentum
nu = np.zeros_like(w)

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X1.shape[0], batch_size)
    loss[i] = compute_loss(X1, y, w)
    #if i % 10 == 0:
    #    visualize(X1[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    nu = alpha*nu + eta*compute_grad(X1[ind,:],y[ind],w-alpha*nu)
    w = w - nu

visualize(X1, y, w, loss)
plt.clf()

print(w)

# ----------------------------------------------------AdaGrad-----------------------------------------------------
print("----------------------AdaGrad -----------------------------------------")
# please use np.random.seed(42), eta=0.05, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])
print(w)

eps = 0.05 #1e-8
eta = 0.8 # learning rate
#alpha = 0.9 # momentum
#nu = np.zeros_like(w)
G = np.zeros_like(w)

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X1.shape[0], batch_size)
    loss[i] = compute_loss(X1, y, w)
    #if i % 10 == 0:
    #    visualize(X1[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    #nu = alpha*nu + eta*compute_grad(X1[ind,:],y[ind],w)
    #w = w - nu
    g = compute_grad(X1[ind,:],y[ind],w)
    for j in range(len(w)):
        G[j] = G[j] + g[j]**2
        w[j] = w[j] - (eta/(np.sqrt(G[j]+eps)))*g[j]

visualize(X1, y, w, loss)
plt.clf()

print(w)


#------------------------------------------------------- RMSPROP --------------------------------------------------------
print("--------------------RMSPROP -------------------------------")
# please use np.random.seed(42), eta=0.1, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)

w = np.array([0, 0, 0, 0, 0, 1.])
print(w)
eta = 0.01 # learning rate
alpha = 0.9 # moving average of gradient norm squared
#g2 = None
eps = 0.05 #1e-8

G = np.zeros_like(w)

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12,5))
for i in range(n_iter):
    ind = np.random.choice(X1.shape[0], batch_size)
    loss[i] = compute_loss(X1, y, w)
    #if i % 10 == 0:
    #    visualize(X1[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    g = compute_grad(X1[ind,:],y[ind],w)
    for j in range(len(w)):
        G[j] = alpha*G[j] + (1-alpha)*g[j]**2
        w[j] = w[j] - (eta/(np.sqrt(G[j]+eps)))*g[j]

visualize(X, y, w, loss)
plt.clf()

print(w)


#------------------------------------------------------- ADAM --------------------------------------------------------
print("Adam -----------------------------------------")
# please use np.random.seed(42), eta=0.1, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)

w = np.array([0, 0, 0, 0, 0, 1.])
print(w)

eta = 0.1 # learning rate
alpha = 0.9 # moving average of gradient norm squared
g2 = None
eps = 0.01 #1e-8

m = np.zeros_like(w)
v = np.zeros_like(w)

beta1 = 0.9
beta2 = 0.9

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12,5))
for i in range(n_iter):
    ind = np.random.choice(X1.shape[0], batch_size)
    loss[i] = compute_loss(X1, y, w)
    #if i % 10 == 0:
    #    visualize(X1[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    g = compute_grad(X1[ind,:],y[ind],w)
    for j in range(len(w)):
        m[j] = (beta1*m[j] + (1-beta1)*g[j] )/(1-beta1**n_iter)
        v[j] = (beta2*v[j] + (1-beta2)*g[j]**2 ) /(1-beta2**n_iter)
        w[j] = w[j] - (eta/(np.sqrt(v[j]+eps)))*m[j]

visualize(X, y, w, loss)
plt.clf()

print(w)








