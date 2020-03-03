import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

frameToSec = lambda frame: frame / 2.67

def frameToSecDF(df):
    newColNames = {}
    for frame in list(df):
        newColNames[frame] = frameToSec(int(frame))
    return df.rename(columns=newColNames, inplace=False)

def removeLowSTD(df, std_multiplier_threshold, t11 = 20, t12 = 31, t21 = 60, t22 = 200):
    base = df[[c for c in list(df) if ((int(c) > 20 and int(c) < 31) or (int(c) > 60 and int(c) < t22))]].copy()
    bstd = base.std(axis=1)
    bmean = base.mean(axis=1)
    keep = set()
    for row in list(df.index.values):
        if row in keep:
            continue
        for col in list(df):
            if (df[col][row] > bmean[row] + bstd[row] * std_multiplier_threshold or
               df[col][row] < bmean[row] - bstd[row] * std_multiplier_threshold):
                keep.add(row)
                break
    k = list(keep)
    return df.loc[k].copy()