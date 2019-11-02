""" 
Back Propagation Neural Network
"""
import numpy as np

class BackpropNN:

    # Constructor of the class
    def __init__(self, in_sz, hid_sz, out_sz, slope):
        # initialize random initial 1weights to the network
        self.Wi = np.random.uniform(size=(hid_sz, in_sz))       # Input layer weights
        self.bi = np.random.uniform(size=(hid_sz, 1))           # Input layer bias
        self.Wh = np.random.uniform(size=(out_sz, hid_sz))      # Hidden layer weights
        self.bh = np.random.uniform(size=(out_sz, 1))           # Hidden layer bias
        self.s = slope                                          # slope of sigmoid function
    
    # Feed forward function 
    def forward(self, x):
        self.zi = np.dot(self.Wi, x) + self.bi
        self.a = self.sigmoid(self.zi)
        self.zh = np.dot(self.Wh, self.a) + self.bh
        return self.sigmoid(self.zh)

    # Derivative of the sigmoid function
    def sig_derivative(self, x):
        return self.s*np.exp((-self.s)*x)/((1+np.exp((-self.s)*x))**2)

    # Sigmoid function
    def sigmoid(self, x):
        return 1/(1+np.exp((-self.s)*x))

    # Loss function
    def loss(self, x, y):
        return (np.sum(x - y)**2)/x.shape[0]

    # Function to train the neural network
    def train(self, data, learning_rate, epochs=1):
        for epoch in range(epochs):
            
            # initialize weights, bias, number of training pairs, loss
            dWh, dbh, dWi, dbi, m, l = np.zeros(self.Wh.shape), np.zeros(self.bh.shape), np.zeros(self.Wi.shape), np.zeros(self.bi.shape), len(data), 0
            
            # Update weights and bias of input to hidden and hidden to output layer
            for (x, y) in data:
                y_hat = self.forward(x)
                l += self.loss(y_hat, y)
                dzh = 2*(y_hat - y)*self.sig_derivative(self.zh)
                dWh += dzh*self.a.T
                dbh += dzh
                da = np.sum(dzh*self.Wh, axis=0).reshape((-1, 1))
                dzi = self.sig_derivative(self.zi)
                dWi += dzi*x.T
                dbi += dzi
            l /= m
            dWh /= m
            dbh /= m
            dWi /= m
            dbi /= m
            self.Wi -= learning_rate*dWi
            self.bi -= learning_rate*dbi
            self.Wh -= learning_rate*dWh
            self.bh -= learning_rate*dbh
            
if __name__ == '__main__':

    ch = int(input('Enter Choice: \n 1. Default Training Set \n 2. User-defined Training Set'))

    if ch == 1:
        # Default Training Data 
        data = [(np.array([[1],[-1]]),np.array([[1]])),(np.array([[-1],[1]]),np.array([[1]])),(np.array([[1],[1]]),np.array([[-1]])),(np.array([[-1],[-1]]),np.array([[-1]]))]
        n = 2       # Input nodes (index j)
        h = 2       # Hidden nodes (index q)
        m = 1       # Output nodes (index k)
        s = 1       # Slope
        eta = 0.5   # Learning rate

    elif ch == 2:
        # User-defined Training Data
        print('\n Enter the Training parameters:')
        n = 2       # Input nodes (index j)
        h = 2       # Hidden nodes (index q)
        m = 1       # Output nodes (index k)
        s = float(input('Slope              : '))
        eta = float(input('Learning Rate      : '))   

    else:
        # Default Training Data 
        data = [(np.array([[1],[-1]]),np.array([[1]])),(np.array([[-1],[1]]),np.array([[1]])),(np.array([[1],[1]]),np.array([[-1]])),(np.array([[-1],[-1]]),np.array([[-1]]))]
        n = 2       # Input nodes (index j)
        h = 2       # Hidden nodes (index q)
        m = 1       # Output nodes (index k)
        s = 1       # Slope
        eta = 0.5   # Learning rate
    
    epochs=int(input('No. of epochs: '))

    NN = BackpropNN(n,h,m,s)        # initialize object
    NN.train(data,eta,epochs)       # train the data

    cases=int(input('\n Enter number of test cases:'))

    for i in range(cases):

        # Create input test list
        test = np.empty(n)
        print('Enter test case input')
        for j in range(n):
            test[j]=float(input())
        # Convert list to numpy array
        test=np.array(test)
        out = NN.forward([(np.array([[test[0]],[test[1]]]))])
        o=out[0,0]
        print('Output: ')
        print(o)