""" 
Boltzman Machine for Travelling Salesman Problem
Oliver-30 Data-Set
"""

import numpy as np
import math
import random

def distanceMatrix(n):
    data = [(54,67),(54,62),(37,84),(41,94),(2,99),(7,64),(25,62),(22,60),(18,54),(4,50),
        (13,40),(18,40),(24,42),(25,38),(44,35),(41,26),(45,21),(58,35),(62,32),(82,7),
        (91,38),(83,46),(71,44),(64,60),(68,58),(83,69),(87,76),(74,78),(71,71),(58,69)]

    dist = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            dist[i][j] = math.sqrt( (data[i][0] - data[j][0] ) ** 2 + ( data[i][1] - data[j][1] ) ** 2 )
    return dist

# Sigmoid activation function
def sigmoid (x):
    return 1 / ( 1 + np.exp( -x / temperature ))

# Calculation of Delta Consensus and Probability of acceptance of the change
def stateProbability (city, tour):
    weights = 0
    for i in range(num):
        if i!=tour:
            weights += state[city][i]*penalty
    for i in range(num):
        if i!=city:
            weights += state[i][tour]*penalty
            if tour == 0:
                weights += state[i][num-1] * (-distanceMatrix[i][city])
                weights += state[i][tour+1] * (-distanceMatrix[i][city])
            elif tour == num-1:
                weights += state[i][tour-1] * (-distanceMatrix[i][city])
                weights += state[i][0] * (-distanceMatrix[i][city])
            else:
                weights += state[i][tour-1] * (-distanceMatrix[i][city])
                weights += state[i][tour+1] * (-distanceMatrix[i][city])
    weights += bias
    deltaConsensus = (1 - (2 * state[city][tour])) * weights

    probability = sigmoid(deltaConsensus)
    return probability, deltaConsensus

#Solving TSP
def solve(temperature):
    R = np.random.uniform(0.5,1)
    while(temperature>0.1):
        for currentCity in range(num**2):
            city = random.randint(0, num-1)
            tour = random.randint(0, num-1)
            A , deltaConsensus = stateProbability(city,tour)
            if(A > R):
                state[city][tour] = 1 - state[city][tour] 
                updations += 1
            else:
                pass       
        temperature *= 0.95
        print('\n Temperature:', temperature, '\t Del_consensus:', deltaConsensus)

def pathDistance(distanceMatrix, path):
    # Calculating path and shortest distance from state matrix
    pathdist = 0
    for i in range(num):
        if(state[i][0] == 1):
            startCity = i
    nextCity = startCity
    path[0] = startCity+1
    for i in range(1,num):
        for j in range(num):
            if(state[j][i] == 1):
                pathdist += distanceMatrix[nextCity][j]
                nextCity = j
                pass
    pathdist += distanceMatrix[nextCity][startCity]
    for j in range(num):
        for i in range(num):
            if(state[i][j] == 1):
                path[j] = i + 1
    return pathdist

if __name__ == "__main__":

    n = int(input(" \n Enter number of cities you wish to run in the Oliver-30 Data-Set: "))
    distanceMatrix = distanceMatrix(n)

    # Print the distance matrix
    print ("  Distance Matrix  ")
    print (distanceMatrix)

    num = np.matrix(distanceMatrix).shape[0]

    # Initialize the parameters
    state = np.eye(num)
    path = np.zeros(num)
    bias = 240
    penalty = -260
    temperature = (-1*penalty - bias)*30

    solve(temperature)
    
    pathdist = pathDistance(distanceMatrix, path)

    # Print distance
    print("\n Optimal Distance = ", pathdist)
    