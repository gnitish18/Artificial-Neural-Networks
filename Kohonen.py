""" 
Kohonen Neural Network
"""
import numpy as np


# Main function
if __name__ == '__main__':

    ch = int(input('Enter Choice: \n 1. Default Training Set \n 2. User-defined Training Set \n'))

    if ch == 1:
        # Default Training Data 
        inputs = np.array([[1,1,1,-1],[-1,-1,-1,1],[1,-1,-1,-1],[-1,-1,1,1]])
        p = 4       # Training Pairs
        n = 4       # Input nodes (index j)
        m = 2       # Output nodes (index k)
        eta = 0.9   # Learning rate
        r = 2.0     # Topological Parameter
        # Weights
        W = np.array([[0.2,0.8],[0.6,0.4],[0.5,0.7],[0.9,0.3]])

    elif ch == 2:
        # User-defined Training Data
        print('\n Enter the Training data:')
        p = int(input('Number of Training pairs     : '))
        n = int(input('Number of Input nodes        : '))
        m = int(input('Number of Output nodes       : '))
        eta = float(input('Learning Rate                : '))
        r = float(input('Topological Parameter        : '))
        inputs = np.empty((p,n))
        W = np.empty((n,m))
        print('\n Enter the input data:')
        for i in range(p):
            for j in range(n):
                inputs[i,j]=(float(input()))
        print('\n Enter the weights of hidden to output layer:')
        for j in range(n):
            for k in range(m):
                W[j,k]=(float(input()))    

    else:
        print('Invalid Input, choosing default training set. \n Press any key to continue... ')
        input()
        # Default Training Data 
        inputs = np.array([[1,1,1,-1],[-1,-1,-1,1],[1,-1,-1,-1],[-1,-1,1,1]])
        p = 4       # Training Pairs
        n = 4       # Input nodes (index j)
        m = 2       # Output nodes (index k)
        eta = 0.9   # Learning rate
        r = 2.0     # Topological Parameter
        # Weights
        W = np.array([[0.2,0.8],[0.6,0.4],[0.5,0.7],[0.9,0.3]])
    
    # Initialize Euclidean Distance
    ED = np.empty((m))
    
    print('Initial Weights:')
    print(W)

    epochs=int(input('No. of epochs: '))

    for iter in range(epochs):
        for i in range(p):
            I=inputs[i,]                    # Seperate individual training pairs
            for k in range(m):
                ED[k]=0
                for h in range(n):
                    ED[k] += (W[h,k]-I[h])**2       # Compute Euclidean distance
            winner = np.argwhere(ED==np.amin(ED))   # Find the winner 
            index = np.size(winner)                 # Find index of winner
            for l in range(index):
                for h in range(n):
                    W[h,winner[l,0]] += eta * (I[h] - W[h,winner[l,0]])   # Update weights
        eta /= r        # Update learning rate
    print('\n Final Weights: ')
    print(W)
    
    # Test cases

    cases=int(input('\n Enter number of test cases:'))
    ED = np.empty((m))
    for i in range(cases):
        # Create input test list
        test = []
        print('Enter test case input')
        for j in range(n):
            test.append(int(input()))
        # Convert list to numpy array
        test = np.array(test)
        # Classify into clusters
        for k in range(m):
            ED[k]=0
            for h in range(n):
                ED[k] += (W[h,k]-test[h])**2
        winner = np.argwhere(ED==np.amin(ED))
        index = np.size(winner)

        print('Cluster:')
        print(winner[0]+1)