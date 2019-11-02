""" 
Radial Basis Function for Curve Fitting
Curves and Contours
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class Radial_Basis(object):

    def __init__(self, hidden_shape, sigma=1.0):
        # radial basis function network
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center - data_point)**2)

    def _calculate_interpolation_matrix(self, X):
        #Calculates interpolation matrix using a kernel_function
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def train(self, X, Y):
        # Fits weights using linear regression
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

def Curve(ch_curve):
    # Generating data for one dimensional curves
    x = np.linspace(0, 10, 100)
    if ch_curve == 1:
        y = np.sin(x)
    elif ch_curve == 2:
        y = np.exp(x)
    elif ch_curve == 3:
        y = x**5 - 4*x**4 + 7*x
    elif ch_curve == 4:
        y = 1/(1+x**2)
    elif ch_curve == 5:
        y = np.exp(-x)*np.sin(x)

    # Fitting the network with data
    model = Radial_Basis(hidden_shape=10, sigma=1.)
    model.train(x, y)
    y_pred = model.predict(x)

    # Plotting the curves
    plt.plot(x, y, 'b-', label='Original Function')
    plt.plot(x, y_pred, 'r-', label='Radial Basis Fit')
    plt.legend(loc='upper right')
    plt.title('Radial Basis Function - 1D interpolation')
    plt.show()

def Contour(ch_contour):
    # Generating contour data for interpolation
    x, y = np.meshgrid(np.linspace(-5, 3, 20), np.linspace(-5, 3, 20))
    if ch_contour == 1:
        z = (np.sin(np.sqrt((x - 2.)**2 + (y - 1)**2)) - np.sin(np.sqrt((x + 2.)**2 + (y + 4)**2))) / 2.
    elif ch_contour == 2:
        z = x**2 - y**2
    elif ch_contour == 3:
        z = np.sin (np.sqrt (x**2 + y**2)) / (np.sqrt (x**2 + y**2))
    elif ch_contour == 4:
        z = np.sin(10*(x**2+y**2))/10.
    elif ch_contour == 5:
        z = x**3 - 5*y

    # Fitting the network with data
    features = np.asarray(list(zip(x.flatten(), y.flatten())))
    model = Radial_Basis(hidden_shape=70, sigma=1.)
    model.train(features, z.flatten())
    predictions = model.predict(features)

    # Plotting the contours
    """
    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    figure.suptitle('Radial Basis Function - Contour Plot', fontsize=20)
    axis_right.set_title('Radial Basis Fit', fontsize=14)
    axis_left.set_title('Original Function', fontsize=14)
    axis_left.contourf(x, y, z)
    right_image = axis_right.contourf(x, y, predictions.reshape(20, 20))
    plt.show()"""


    fig = go.Figure(data=[go.Surface(z=z)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
    fig1 = go.Figure(data=[go.Surface(z=predictions.reshape(20,20), showscale=False, opacity=0.9)])
    fig1.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
    #fig = go.Figure(data=[go.Surface(z=z),go.Surface(z=predictions.reshape(20,20)+5, showscale=False, opacity=0.9)])
    
    #go.Surface(z=z, showscale=False, opacity=0.9)
    #])
    fig.show()
    fig1.show()

if __name__ == "__main__":

    #choice = int(input(" \n Enter the data set you wish to train: \n 1. One dimensional Curves \n 2. Contour Plots \n"))
    """if choice == 1:
        ch_curve = int(input(" \n Choose the curve you wish to train: \n 1. Trigonometric \n 2. Exponential \n 3. Polynomial \n 4. Rational \n 5. Trig-Exponential \n 6. Print all \n"))
        if ch_curve == 6:
            for i in range(5):
                Curve(i+1)
        else:
            Curve(ch_curve)
    """
    if 1:
        ch_contour = int(input(" \n Choose the contour you wish to train: \n 1. Trigonometric \n 2. Saddle \n 3. Sombrero \n 4. Ripple \n 5. Polynomial \n 6. Print all \n"))
        if ch_contour == 6:
            for i in range(5):
                Contour(i+1)
        else:
            Contour(ch_contour)