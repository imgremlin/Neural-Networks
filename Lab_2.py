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
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
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
        #print (self.weights)

    def predict(self, X):

        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions
    
    
# generating data
x = np.linspace(-2, 5, 50)
y = x**3-5*x**2-10*x
#y_u = y + 30
#y_d = y - 30
#x = np.linspace(2, 14, 100)
#y = np.cos(x / 5) * np.sin(x / 10) + 5 * np.exp(-x / 2)

# fitting RBF-Network with data
model = RBFN(hidden_shape=50, sigma=5)
model.fit(x, y)
y_pred = model.predict(x)

# plotting 1D interpolation
plt.plot(x, y, 'b-', label='real')
plt.plot(x, y_pred, 'r-', label='fit', marker='.')
#plt.plot(x, y_u, linestyle=':')
#plt.plot(x, y_d, linestyle=':')
plt.margins(0.05)

X1 = []
Y1 = []
X2 = []
Y2 = []
for i in np.arange(-2, 5, 0.05):
    f = i**3-5*i**2-10*i
    Y1.append(np.random.uniform(f+5, f+45))
    X1.append(i)
    Y2.append(np.random.uniform(f-45, f-5))
    X2.append(i)
plt.scatter(X1, Y1, color='c', label='1', s=20, alpha=0.5)
plt.scatter(X2, Y2, color='y', label='0', s=20, alpha=0.5)
plt.title('Interpolation using a RBFN (Fit)')
plt.legend(loc=3)
plt.show()

plt.plot(x, y_pred, 'r-', label='test', marker='.')
#plt.plot(x, y_u, linestyle=':')
#plt.plot(x, y_d, linestyle=':')
plt.margins(0.05)

X1 = []
Y1 = []
X2 = []
Y2 = []
for i in np.arange(-2, 5, 0.2):
    f = i**3-5*i**2-10*i
    Y1.append(np.random.uniform(f-2, f+50))
    X1.append(i)
    Y2.append(np.random.uniform(f-45, f-5))
    X2.append(i)
plt.scatter(X1, Y1, color='c', label='1', s=30)
plt.scatter(X2, Y2, color='y', label='0', s=30)
plt.title('Interpolation using a RBFN (Test)')
plt.legend(loc=3)
plt.show()

err = []
for i in range(10):
    #error = np.exp(-(i)**2)
    error = ((y_pred[i] - (x[i]**3-5*x[i]**2-10*x[i]))**2)
    err.append(error)
plt.plot(err)
plt.title('Error of function')
plt.show()
