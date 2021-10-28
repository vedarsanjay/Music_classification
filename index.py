from python_speech_features import mfcc #Mel Frequency Copstral Coefficient
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile

import os
import pickle
import random
import operator
import math

# function to get the distance between feature vectors and find neighbors
def getNeighbors(trainingSet, instance, k):
    distance = [] #distance list
    #calculates all possible distance among items or instance
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distance.append((trainingSet[x][2], dist))

    #sorting the data   
    distance.sort(key=operator.itemgetter(1))
    #for k nearest neighbors
    neighbors = [] #new list
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors

# identify the class of the neighbors
def nearestClass(neighbors):
    classVote = {} #set or dictionary

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key= operator.itemgetter(1), reverse=True)

    return sorter[0][0] #sorted array

# function to evaluate the model
def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    return (1.0 * correct) / len(testSet) 

# directory that hold the dataset
directory = "./myproject/Data/genres"

f = open("my.dat", 'wb')
i = 0
# considering 10 neighbors
for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+folder):
        (rate, sig) = wav.read(directory+folder+"/"+file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)
        pickle.dump(feature, f)

f.close()

dataset = []
def loadDataset(filenam, split, trSet, teSet):
    with open("my.dat", rb) as f:
        while True:
           try:
               dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])

trainingSet = []
testSet = []
loadDataset("my.dat", 0.66, trainingSet, testSet)




