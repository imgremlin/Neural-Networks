import numpy as np
import matplotlib.pyplot as plt

class RBFN(object):

    def __init__(self, hidden_shape, sigma=1.0):  
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def fit(self, X, Y):
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)
        print('massive of weights:')
        for i in range(0,self.weights.size):
            print('w[{}] = {}'.format(i, round(self.weights[i], 3)))

    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions
    
x = np.linspace(-5, 5, 100)
a = 3
y = (a**3)/(x**2 + a**2)

model = RBFN(hidden_shape=25, sigma=2)
model.fit(x, y)
y_pred = model.predict(x)

plt.plot(x, y,  label='real')
plt.plot(x, y_pred, label='rbf interpolation')
plt.legend()
plt.title('RBF')
plt.show()
