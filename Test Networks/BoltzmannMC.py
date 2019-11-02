""" 
Boltzman Machine for Travelling Salesman Problem
Oliver-30 Data-Set
"""

import numpy as np
import sys
import itertools
import math
import time
import collections

min_distance = []

def distanceMatrix(n):
    data = [(54,67),(54,62),(37,84),(41,94),(2,99),(7,64),(25,62),(22,60),(18,54),(4,50),
        (13,40),(18,40),(24,42),(25,38),(44,35),(41,26),(45,21),(58,35),(62,32),(82,7),
        (91,38),(83,46),(71,44),(64,60),(68,58),(83,69),(87,76),(74,78),(71,71),(58,69)]

    dist = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            dist[i][j] = math.sqrt( (data[i][0] - data[j][0] ) ** 2 + ( data[i][1] - data[j][1] ) ** 2 )
    return np.matrix(dist)

def path(state):
    path = []
    for j in range(state.shape[1]):
        path.append(np.argmax(state[:,j]))
    return path

def pathDistance(distanceMatrix, path):
    distance = 0
    for index in range(len(path))[1:]:
        distance += distanceMatrix[path[index - 1], path[index]]
    return distance

def Path_Validity(state):
    # Count and update visits
    duplicateVisits = 0
    k = None

    # Check validity of visits
    for i in range(state.shape[0]):
        timesVisited = np.sum(state[i, :])

        if timesVisited != 1 and timesVisited != 2:
            return False

        if timesVisited == 2:
            duplicateVisits += 1
            k = i

    # Check for duplicate visits
    if duplicateVisits != 1:
        return False
    if state[k,0] != 1 or state[k,-1] != 1:
        return False
    for j in range(state.shape[1]):
        citiesVisitedAtOnce = np.sum(state[:, j])
        if citiesVisitedAtOnce != 1:
            return False

    return True

class TSP:

    # Constructor
    def __init__(self, distanceMatrix):

        self.distanceMatrix = distanceMatrix
        self.numCities = distanceMatrix.shape[0]
        self.tourSteps = self.numCities + 1
        self.numStates = self.numCities * self.tourSteps

        self.states = np.eye(self.tourSteps)
        self.states = np.delete(self.states, self.numCities, axis=0)

        # Constraint: penalty > bias >= 2 * longestDistance
        bias = 2*np.max(self.distanceMatrix)
        penalty = 2*bias

        self.weights = self._initWeights(penalty, bias)
        self.temperature = self._initTemp(penalty, bias)

    def _initWeights(self, penalty, bias):
        # Initialize Weights
        weights = np.zeros( (self.numCities, self.tourSteps, self.numCities, self.tourSteps) )

        # Check for impossible configurations
        for city in range(self.numCities):
            distances = self.distanceMatrix[city, :]
            for tourStep in range(self.numCities+1):
                curWeights = weights[city, tourStep]

                prevTourStep = tourStep - 1 if tourStep > 0 else self.tourSteps - 1
                curWeights[:, prevTourStep] = distances

                nextTourStep = tourStep + 1 if tourStep < self.tourSteps - 1 else 0
                curWeights[:, nextTourStep] = distances

                curWeights[:, tourStep] = penalty

                if tourStep == 0:
                    curWeights[city, :-1] = penalty
                elif tourStep == self.numCities:
                    curWeights[city, 1:] = penalty
                else:
                    curWeights[city, :] = penalty

                if tourStep == 0 or tourStep == self.numCities:
                    curWeights[city, 0] = -bias
                    curWeights[city, self.numCities] = -bias
                else:
                    curWeights[city, tourStep] = -bias
        return weights

    def _initTemp(self, penalty, bias):
        # Initialising temperature
        o = ((penalty * self.numCities * self.tourSteps) - bias) * 100
        n = ((penalty) - bias)
        return n

    def _stateProbability(self, city, tour, temperature):
        # DELTA Consensus

        states = self.states.copy()

        # Weights and current state for the given city/tour step
        state = self.states[city, tour]
        weights = self.weights[city, tour]

        # Flipping the state at the given city/tour
        states[city, tour] = (1 - state)

        # Calculating the activity value
        weightEffect = np.sum(weights * states)
        biasEffect = weights[city, tour]
        activityValue = weightEffect + biasEffect

        deltaConsensus = ((1 - state) - state) * activityValue

        # Sigmoid activation function
        exponential = np.exp(-1 * deltaConsensus / temperature)
        probability = 1 / (1 + exponential)

        return probability, deltaConsensus

    def solve(self):
        lastValidState = self.states.copy()
        lowest_temp = 0.1
        shortStart = None
        validHits = 0
        statesExplored = collections.defaultdict(int)
        changes = 0

        # Run until stopping condition (temperature = 0)
        while self.temperature > lowest_temp:

            if shortStart == None:
                shortStart = time.time()

            # Exploring states
            for _ in range(self.numStates**2):

                # Select a random city and tour
                city = np.random.randint(0, self.numCities-1, 1)[0]
                tour = np.random.randint(0, self.tourSteps-1, 1)[0]

                stateProbability, deltaConsensus = self._stateProbability(city, tour, self.temperature)

                # Randomly flip the state based on the state probability
                if np.random.binomial(1, stateProbability) == 0:
                    changes += 1 

                    self.states[city, tour] = 1 - self.states[city, tour]

                    # Flip the first or last state based on previous flip
                    if tour == 0:
                        self.states[city, self.tourSteps-1] = self.states[city, tour]
                    elif tour == self.tourSteps-1:
                        self.states[city, 0] = self.states[city, tour]

                    if Path_Validity(self.states):
                        lastValidState = self.states.copy()
                        statesExplored[str(lastValidState)] += 1
                        validHits += 1

            # Temperature update
            self.temperature *= 0.95

            if time.time()-shortStart > 1:
                shortStart = None
                dist = pathDistance(self.distanceMatrix, path(lastValidState))
                min_distance.append(dist)
                print("Temperature:%-12s  Updations:%-10s  Del_Consensus:%-10s " % ( "%.2f"%self.temperature, str(changes), str(deltaConsensus) ))
                
                changes = 0

        return path(lastValidState)

def show(paths, dist):
    # Print path and distance
    print (" Optimal path (Distance=%d):" % (dist))
    for path in sorted(paths):
        print("   ", path)

if __name__ == "__main__":

    n = int(input(" \n Enter number of cities you wish to run in the Oliver-30 Data-Set: "))
    distanceMatrix = distanceMatrix(n)

    # Print the distance matrix
    print ("  Distance Matrix  ")
    print ("  ",str(distanceMatrix).replace("\n","\n   "))

    # Initialize the class object
    tsp = TSP(distanceMatrix)
    path = tsp.solve()
    dist = pathDistance(distanceMatrix, path)
    # Print path and distance
    show([path],  min(min_distance))
