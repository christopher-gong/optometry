import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from importHelpers.cluster import *

def combine(initial, n, nc = 7, std_thresh = 20, output = False):
    ''' Runs the PCA n times on a normalized initial and returns a dataframe with the results combined. '''
    normed = pd.DataFrame(StandardScaler().fit_transform(initial.loc[:, list(initial)].values))
    normed.rename(index={key:value for (key,value) in zip(list(normed.index.values), list(initial.index.values))}, inplace=True)
    return runmerge(normed, 10)

############## HELPERS ##############

def toLists(d):
    z = []
    k = list(d.keys())
    for i in range(len(d)):
        z += [[0] * len(d)]
        for j in range(len(d)):
            if (j < i + 1):
                z[i][j] = 0
            elif (d[k[i]] == d[k[j]]):
                z[i][j] = 1
    return z

def listToDF(lsts, k):
    df = pd.DataFrame(lsts, columns=k)
    df['x'] = k
    df.set_index('x', inplace=True)
    #df.rename(index={0:'zero',1:'one'}, inplace=True)
    return df

def dictToDF(d):
    return listToDF(toLists(d), d.keys())

def mergeProcedure(listDict):
    assert len(listDict) > 1, "You must merge multiple dictionaries."
    mergedLists = toLists(listDict[0])
    #print(dictToDF(listDict[0]))
    for z in range(1, len(listDict)):
        mult = 1 / (z + 1)
        d = listDict[z]
        l = toLists(d)
        #print(dictToDF(d))
        for i in range(len(listDict[0])):
            for j in range(i + 1, len(listDict[0])):
                mergedLists[i][j] = (1 - mult) * mergedLists[i][j] + (mult) * l[i][j]
            #print("Merge ", i)
            #print(listToDF(mergedLists, listDict[0].keys()))
    return mergedLists

def runmerge(initial, n):
    #out_df, _ = cluster(initial, nc = 25, std_thresh = 2, output = False)
    #out_df, _ = cluster(initial, output = False)
    l = []
    for _ in range(n):
        out_df, c = cluster(initial, output = False)
        d = out_df.set_index("in").to_dict()['out']
        l += [d]
    return listToDF(mergeProcedure(l), list(initial.index.values))
