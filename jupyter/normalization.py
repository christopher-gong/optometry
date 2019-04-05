import csv
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats.mstats import zscore
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pylab import *
import fits
import copy
import numbers

def determineThresholds(col, table, cutoff=0.01): #change bins?
    '''Determines the thresholds for the column of the table with the given cutoff.
    Returns the left x and right x that at the cutoff value. For example, with a cutoff of
    0.01, this fn will return the left x and right x that keeps 1% of the mass on the left
    and 1% of the mass on the right, respectively.'''
    assert(cutoff > 0 and cutoff < 1)
    nparr = np.asarray(table[col].tolist())
    y,x,_=hist(nparr,1000,alpha=.3,label='data')
    x=(x[1:]+x[:-1])/2
    sum = 0
    for ycord in y:
        sum += ycord
    left = sum * cutoff
    right = sum * (1 - cutoff)
    sum = 0
    leftx = 0
    rightx = 0
    for xcord, ycord in zip(x,y):
        sum += ycord
        if (left < sum and leftx == 0):
            leftx = xcord
        if (right < sum and rightx == 0):
            rightx = xcord
            break
    return leftx, rightx

def normalize(col, table, cutoff=0.01):
    '''Determines thresholds using the determineThresholds fn. For all numbers
    greater than 1, they are made 1. For all numbers less than 0, they are made 0.
    A new col named normalized + colname is created in the table.'''
    leftx, rightx = determineThresholds(col, table, cutoff=0.01)
    def new(signal):
        normsignal = (signal - leftx) / (rightx - leftx)
        if (normsignal > 1):
            return 1
        elif (normsignal < 0):
            return 0
        else:
            return normsignal
    return table[col].apply(new)

def normalizeDAPI(roi_df):
    ''' normalizeDAPI fn.
    1. normalizes DAPI with the normalize fn of cutoff=0.01 (1% of the mass on left and 1% of mass on right).
    2. removes all rows where the normalized DAPI is less than or equal to 0.2. '''
    try:
        roi_df["norm DAPI"] = normalize("DAPI", roi_df)
        dapi_rm_roi_df = roi_df[roi_df['norm DAPI'] > 0.2]
        return dapi_rm_roi_df
    except:
        print("No DAPI found! DAPI normalization did not occur.")
        return roi_df

def normalizeProcedure(roi_df, unwantedCols=[], quiet=True):
    '''Performs the Normalize Procedure on the table. 
    1. normalizes DAPI with the normalize fn of cutoff=0.01 (1% of the mass on left and 1% of mass on right).
    2. removes all rows where the normalized DAPI is less than or equal to 0.2. 
    3. normalizes and z normalizes the rest of the columns, named norm and z norm (except unwantedCols).
    4. returns the new table. '''
    #Step 1, 2
    dapi_rm_roi_df = normalizeDAPI(roi_df)
    #Step 3
    rm_roi_df = dapi_rm_roi_df.copy()
    unwantedCols = []
    cols = list(dapi_rm_roi_df)
    selectedCols = [c for c in cols if c not in unwantedCols]
    for col in selectedCols:
        if col.startswith("norm") or (dapi_rm_roi_df[col]).isna().any() or not isinstance(list(dapi_rm_roi_df[col])[0], numbers.Real):
            if (not quiet):
                print("Passed ", col)
            continue
        if (not quiet):
            print("Normalizing  ", col)
        rm_roi_df["norm " + col] = normalize(col, dapi_rm_roi_df)
    numeric_cols = rm_roi_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c.startswith("norm")]
    z_roi_df = rm_roi_df[numeric_cols].apply(zscore).add_prefix('z ')
    rm_roi_df = pd.concat([rm_roi_df, z_roi_df], axis=1, sort=False)
    #Step 4
    return rm_roi_df

def normHists(df, col):
    return df[[col, "norm " + col, "z norm " + col]].hist(bins=1000,figsize=(20,10))

def meanSTD(df, cols, index):  
    slideinfo = []
    unwantedCols2=[]
    selectCols = cols
    #edit the item index to change whether the binning occurs across areas or across slides [0:7]
    for slide in set([item[0:index] for item in list(df["image "])]):
        for col in selectCols:
           slidecol = df[df['image '].str.contains(slide)][col]
           slideinfo.append([slide, col, slidecol.mean(), slidecol.std()])
    selectMeanSTD = pd.DataFrame(slideinfo, columns=['slide', 'col', 'mean', 'std'])
    return selectMeanSTD.sort_values("col")