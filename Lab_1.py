import pandas as pd
import numpy as np
import random as rm
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('data25.csv', names=['info'])

res = []
s = []

for i in range(0, 100):
    strk = data['info'][i]
    strk = strk.split(';')

    s.append([])
    s[i].append(strk[0])
    s[i].append(strk[1])
    s[i].append(strk[2])

rm.shuffle(s)

df = []
res = []
for i in range(0,100):
    df.append([])
    df[i].append(s[i][0])
    df[i].append(s[i][1])
    res.append(s[i][2])
    
df = np.array(df).astype(float)
res = np.array(res).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, res, test_size=0.2, random_state=0)

def lin(outcome):
    return outcome

def binary(outcome):
    return np.where(outcome > 0, 1, 0)
    
def relu(outcome):
    return np.where(outcome > 0, outcome, 0)

def sigmoid1(outcome):
    return 1/(math.exp(-outcome) + 1)

def erf(outcome):
    return math.tanh(outcome)

def bent(outcome):
    return (math.sqrt(outcome ** 2 + 1) - 1)/2 + outcome


class RBPerceptron:
  
  

  def __init__(self, number_of_epochs = 100, learning_rate = 0.1):
    self.number_of_epochs = number_of_epochs
    self.learning_rate = learning_rate

  def train(self, X, D):
    # Initialize weights vector with zeroes
    num_features = X.shape[1]
    err = []
    self.w = np.zeros(num_features + 1)
    # Perform the epochs
    for i in range(self.number_of_epochs):
      # For every combination of (X_i, D_i)
      aver = 0
      iter = 0
      for sample, desired_outcome in zip(X, D):
        # Generate prediction and compare with desired outcome
        prediction    = self.predict(sample)
        difference    = (desired_outcome - prediction)
        aver = aver + difference
        # Compute weight update via Perceptron Learning Rule
        weight_update = self.learning_rate * difference
        self.w[1:]    += weight_update * sample
        self.w[0]     += weight_update
        iter = iter + 1
      err.append(math.fabs(aver)/ iter)      
    #print(err)
    plt.plot([x for x in range(self.number_of_epochs)], err)
    plt.title('Errors of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()
    print ('Weights vector: w1 =', round(self.w[1], 4), 'w2 =', round(self.w[2], 4), 'b =', round(self.w[0], 4))
    return self
  
      
  # Generate prediction
  def predict(self, sample):
    outcome = np.dot(sample, self.w[1:]) + self.w[0]
    func_vect = np.vectorize(bent)
    return func_vect(outcome)

rbp = RBPerceptron(800, 0.05)

train_model = rbp.train(X_train, y_train)

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train, y_train, clf=train_model)
plt.title('Perceptron Train')
plt.xlabel('X_train')
plt.ylabel('Y_train')
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.show()

plot_decision_regions(X_test, y_test, clf=train_model)
plt.title('Perceptron Test')
plt.xlabel('X_test')
plt.ylabel('Y_test')
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.show()
