import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('data25.csv', names=['info'])

res = []
df = []

for i in range(0, 100):
    strk = data['info'][i]
    strk = strk.split(';')
    res.append(strk[2])
    df.append([])
    df[i].append(strk[0])
    df[i].append(strk[1])
    
df = np.array(df).astype(float)
res = np.array(res).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, res, test_size=0.2, random_state = 0)

class RBPerceptron:

  def __init__(self, number_of_epochs = 100, learning_rate = 0.1):
    self.number_of_epochs = number_of_epochs
    self.learning_rate = learning_rate

  def train(self, X, D):
      
    num_features = X.shape[1]
    self.w = np.zeros(num_features + 1)
    
    for i in range(self.number_of_epochs):
      for sample, desired_outcome in zip(X, D):
          
        prediction = self.predict(sample)
        difference = (desired_outcome - prediction)
        weight_update = self.learning_rate * difference
        self.w[1:] += weight_update * sample
        self.w[0] += weight_update
    print('massive of weights:', self.w)
    return self

  def predict(self, sample):
    outcome = np.dot(sample, self.w[1:]) + self.w[0]
    return np.where(outcome > 0, 1, 0)

rbp = RBPerceptron(600, 0.1)
trained_model = rbp.train(X_train, y_train)

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_test, y_test, clf=trained_model)
plt.title('Perceptron')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.show()