
import numpy as np
import pandas as pd
#from sklearn import datasets

from sklearn.cross_validation import train_test_split

iris = pd.read_csv("/home/terrence/CODING/Python/MODELS/Datasets/iris/Iris.csv")
#print(iris.columns)
print(iris.head(3))

X1 = iris['SepalLengthCm']
X2 = iris['SepalWidthCm']
X3 = iris['PetalLengthCm']
X4 = iris['PetalWidthCm']
X = pd.concat([X1,X2,X3,X4],axis =1)
print(X.head(3))
y1 = iris.Species
print(np.unique(y1))

y = iris['Species'].map(lambda x : 1 if x == 'Iris-setosa' else 0)
print(y.head(10))
print(np.unique(y))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

def logistic(X, w,bias):
    b = np.dot(X,w) + bias
    c = 1/(1+np.exp(-b))
    return c

w = np.linspace(-1,1,4)
bias = np.linspace(-1,1,len(X_train))
res = logistic(X_train,w,bias)
print(res.shape)
print(res[:10])

'''
def compute_loss(X, y, w):
    a = - np.sum(np.multiply(y, np.log1p(logistic(X,w))) + np.multiply((1-y), np.log1p(1-logistic(X,w))))
    return a

res2 = compute_loss(X_train,y_train,w)
print(res2)


def compute_grad(X, y, w):
    a = y - logistic(X,w)
    res = np.dot(a,X)
    return res
    
res3 = compute_grad(X_train,y_train,w)
print(res3)

#--------------- batch-GD --------------------------------
print("--------------- batch-GD ---------------------")
np.random.seed(42)
w = np.array([0, 0, 0,1])
print(w)
eta= 0.1 # learning rate

n_iter = 3
batch_size = len(X_train)
loss = np.zeros(n_iter)
#plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_train.shape[0], batch_size)
    loss[i] = compute_loss(X_train, y_train, w)
    #if i % 10 == 0:
    #    visualize(X_train[ind, :], y_train[ind], w, loss)

    # TODO:<your code here>
    w = w - eta*compute_grad(X_train[ind,:],y_train[ind],w)

#visualize(X_train, y_train, w, loss)
#plt.clf()
print(w)

y_BGD = np.dot(X_test,w)
print(y_BGD)
print(y_test)
print((y_BGD == y_test).sum())

'''