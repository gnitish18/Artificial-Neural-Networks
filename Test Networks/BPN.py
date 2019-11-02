""" 
Back Propagation Neural Network
"""
import numpy as np
import math

def sigmoid(s,x):
    return 1/(1+math.exp(-s*x))

if __name__ == '__main__':

    ch = int(input('Enter Choice: \n 1. Default Training Set \n 2. User-defined Training Set \n'))

    if ch == 2:
        # User-defined Training Data
        print('\n Enter the Training parameters:')
        # User-defined Training Data
        print('\n Enter the Training data:')
        p = int(input('Number of Training pairs     : '))
        n = int(input('Number of Input nodes        : '))
        h = int(input('Number of Hidden layer nodes : '))
        m = int(input('Number of Output nodes       : '))
        s = float(input('Slope                        : '))
        eta = float(input('Learning Rate                : '))
        inputs = np.empty((p,n))
        outputs = np.empty((p,m))
        Bm = np.empty(m)
        Z = np.empty((h,m))
        Bh = np.empty(h)
        W = np.empty((n,h))
        print('\n Enter the input data:')
        for i in range(p):
            for j in range(n):
                inputs[i,j]=(float(input()))
        print('\n Enter the output data:')
        for i in range(p):
            for k in range(m):
                outputs[i,k]=(float(input()))
        print('\n Enter the bias of hidden to output layer:')
        for k in range(m):
            Bm[k]=(float(input()))
        print('\n Enter the weights of hidden to output layer:')
        for q in range(h):
            for k in range(m):
                Z[q,k]=(float(input()))
        print('\n Enter the bias of input to hidden layer:')
        for q in range(h):
            Bh[q]=(float(input()))
        print('\n Enter the weights of hidden to output layer:')
        for j in range(n):
            for q in range(h):
                W[j,q]=(float(input()))    

    else:
        # Default Training Data 

        inputs = np.array([[0.1,0.2],[0.2,0.3],[0.3,0.4],[0.4,0.5],[0.3,0.5],[0.2,0.4],[0.0,0.0],[0.6,0.4]])
        inputs *= 10

        # output = (x + y) / 10.0
        outputs = np.array([[0.3,0.5,0.7,0.9,0.8,0.6,0.0,1.0]]).T


        """
        inputs = np.zeros((400,2))
        outputs = np.zeros((400,1))
        print('\n Input: ',inputs)
        x, y = np.meshgrid(np.linspace(-5, 3, 20), np.linspace(-5, 3, 20))
        z = x**2 - y**2
        k = 0
        for i in range (20):
            for j in range(20):
                inputs[k,0] = x[i,j]
                inputs[k,1] = y[i,j]
                outputs[k,0] = z[i,j]
                k += 1
        
        print('\n Input: ',inputs, outputs)
        print('\n X : ',np.shape(x))
        print('\n Y : ',np.shape(y))
        print('\n Z : ',np.shape(z))
        input()
        """

        p = 8       # Training Pairs
        
        n = 2       # Input nodes (index j)
        h = 2       # Hidden nodes (index q)
        m = 1       # Output nodes (index k)
        s = 1       # Slope
        eta = 0.5   # Learning rate
        # Weights and bias for output layer
        Bm = np.array([-0.5])
        Z = np.array([[0.5,0.6]]).T
        # Weights and bias for hidden layer
        Bh = np.array([[-0.5,-0.5]]).T
        W = np.array([[0.1,0.2],[0.3,0.4]])
    
    print('\n LAYER 1:')
    print('\n Initial Weights:')
    print(W)
    print('\n Initial Bias:')
    print(Bh)
    print('\n LAYER 2:')
    print('\n Initial Weights:')
    print(Z)
    print('\n Initial Bias:')
    print(Bm)

    epochs=int(input('\n No. of epochs: '))

    error = np.zeros((m,1))

    for iter in range(epochs):
        for i in range(p):

            sumh = np.zeros((h,1))                      # initialize hidden layer sum
            H = np.zeros((h,1))                         # initiatize activated sum

            # Hidden Layer
            for q in range(h): 
                for j in range(n):
                    sumh[q] += (inputs[i,j] * W[j,q])   # Weighted sum
                sumh[q] += Bh[q]                        # Add bias
                H[q]=sigmoid(s,sumh[q])                 # Activation function

            summ=np.zeros((m,1))                        # initialize output layer sum
            O=np.zeros((m,1))                           # initialize activated sum (or) output

            # Output Layer
            for k in range(m):
                for q in range(h):
                    summ[k] += H[q]*Z[q,k]              # Weighted sum
                summ[k] += Bm[k]                        # Add bias
                O[k]=sigmoid(s,summ[k])                          # Purlin Activation function

            for k in range(m):
                error[k] = (outputs[i,k]-O[k])

            for q in range(h):
                for k in range(m):
                    Z[q,k] += 2*eta*s * error[k] * O[k] * (1-O[k]) * H[q,k]
                    Bm[k] += 2*eta*s * error[k] * O[k] * (1-O[k])

            su = 0 
            for j in range(n):
                for q in range(h):
                    for k in range(m):
                        W[j,q] += 2*eta*s**2 * error[k] * O[k] * (1-O[k]) * Z[q,k] * H[j,k] * (1-H[j,k]) * inputs[i,j]
                        Bh[q] += 2*eta*s**2 * error[k] * O[k] * (1-O[k]) * Z[q,k] * H[j,k] * (1-H[j,k])

    cases=int(input('\n Enter number of test cases:'))

    for i in range(cases):

        # Create input test list
        test = np.empty(n)
        print('Enter test case input')
        for j in range(n):
            test[j]=float(input())
        # Convert list to numpy array
        test=np.array(test)

        sumh = np.zeros((h,1))
        H = sumh

        # Calculate hidden layer nodes
        for q in range(h): 
            for j in range(n):
                sumh[q] += (test[j] * W[j,q])   
            sumh[q] += Bh[q]                        
            H[q]=sigmoid(s,sumh[q])                   

        summ=np.zeros((m,1))
        O=np.zeros((m,1))

        # Calculate output layer nodes
        for k in range(m):
            for q in range(h):
                summ[k] += H[q,k]*Z[q,k]
            summ[k] += Bm[k]
            O[k]=sigmoid(s,summ[k])

        print('Output:')
        print(O)