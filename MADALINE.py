""" 
MADALINE Neural Network
"""
import numpy as np

# Defining bipolar function - activation function
def bipolar(x):
    if x >= 0:
        return 1
    else:
        return -1

# Main function
if __name__ == '__main__':

    ch = int(input('Enter Choice: \n 1. Default Training Set \n 2. User-defined Training Set'))

    if ch == 1:
        # Default Training Data 
        inputs = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
        outputs = np.array([[-1,1,1,-1]]).T
        p = 4       # Training Pairs
        n = 2       # Input nodes (index j)
        h = 2       # Hidden nodes (index q)
        m = 1       # Output nodes (index k)
        eta = 0.7   # Learning rate
        # Weights and bias for output layer
        Bm = np.array([0.5])
        Z = np.array([[0.5,0.6]]).T
        # Weights and bias for hidden layer
        Bh = np.array([[0.5,0.5]]).T
        W = np.array([[0.1,0.2],[0.3,0.4]])

    elif ch == 2:
        # User-defined Training Data
        print('\n Enter the Training data:')
        p = int(input('Number of Training pairs     : '))
        n = int(input('Number of Input nodes        : '))
        h = int(input('Number of Hidden layer nodes : '))
        m = int(input('Number of Output nodes       : '))
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
        print('Invalid Input, choosing default training set. \n Press any key to continue... ')
        input()
        # Default Training Data 
        inputs = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
        outputs = np.array([[-1,1,1,-1]]).T
        p = 4       # Training Pairs
        n = 2       # Input nodes (index j)
        h = 2       # Hidden nodes (index q)
        m = 1       # Output nodes (index k)
        eta = 0.7   # Learning rate
        # Weights and bias for output layer
        Bm = np.array([0.5])
        Z = np.array([[0.5,0.6]]).T
        # Weights and bias for hidden layer
        Bh = np.array([[0.5,0.5]]).T
        W = np.array([[0.1,0.2],[0.3,0.4]])

    print('Initial Weights:')
    print(W)
    print('\nInitial Bias:')
    print(Bh)

    epochs=int(input('No. of epochs: '))

    for iter in range(epochs):
        for i in range(p):

            sumh = np.zeros((h,1))                      # initialize hidden layer sum
            H = np.zeros((h,1))                         # initiatize activated sum

            # Hidden Layer
            for q in range(h): 
                for j in range(n):
                    sumh[q] += (inputs[i,j] * W[j,q])   # Weighted sum
                sumh[q] += Bh[q]                        # Add bias
                H[q]=bipolar(sumh[q])                   # Activation function

            summ=np.zeros((m,1))                        # initialize output layer sum
            O=np.zeros((m,1))                           # initialize activated sum (or) output

            # Output Layer
            for k in range(m):
                for q in range(h):
                    summ[k] += H[q,k]*Z[q,k]            # Weighted sum
                summ[k] += Bm[k]                        # Add bias
                O[k]=bipolar(summ[k])                   # Activation function

            # Checking for Weight Updation
            for k in range(m):
                # output and target vectors (or) scalars are same
                if outputs[i,k]==O[k]:
                    pass
            for k in range(m):
                if outputs[i,k] != O[k]:
                    if outputs[i,k]==1:
                        dist = (sumh - O)**2                    # initialize difference or distance
                        Q=np.argwhere(dist==np.amin(dist))      # find index of minimum difference
                        for j in range(n):
                            W[j,Q[0,0]] += eta * (1 - sumh[Q[0,0]]) * inputs[i,j]   # Update weights
                        Bh[Q[0,0]] += eta * (1 - sumh[Q[0,0]])                      # Update bias
                if outputs[i,k] != O[k]:
                    if outputs[i,k]==-1:
                        zero = np.zeros((m,1))                  # initialize zero vector for checking
                        R =np.argwhere(sumh>zero)               # find index of positive sum
                        rz=len(R)    
                        for r in range(rz):
                            for j in range(n):
                                W[j,R[r,0]] += eta * (-1 - sumh[R[r,0]]) * inputs[i,j]  # Update weights
                            Bh[R[r,0]] += eta * (-1 - sumh[R[r,0]])                     # Update bias
    print('\n Final Weights: ')
    print(W)
    print('\n Final Bias:   ')
    print(Bh)

    # Test cases

    cases=int(input('\n Enter number of test cases:'))

    for i in range(cases):

        # Create input test list
        test = []
        print('Enter test case input')
        for j in range(n):
            test.append(int(input()))
        # Convert list to numpy array
        test=np.array(test)

        sumh = np.zeros((h,1))
        H = sumh

        # Calculate hidden layer nodes
        for q in range(h): 
            for j in range(n):
                sumh[q] += (test[j] * W[j,q])   
            sumh[q] += Bh[q]                        
            H[q]=bipolar(sumh[q])                   

        summ=np.zeros((m,1))
        O=np.zeros((m,1))

        # Calculate output layer nodes
        for k in range(m):
            for q in range(h):
                summ[k] += H[q,k]*Z[q,k]
            summ[k] += Bm[k]
            O[k]=bipolar(summ[k])

        print('Output:')
        print(O)

