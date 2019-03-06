import csv
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pylab import *

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def gaussGraph(col, table, expected=None):
    nparr = np.asarray(table[col].tolist())
    y,x,_=hist(nparr,100,alpha=.3,label='data')
    x=(x[1:]+x[:-1])/2
    if expected is None:
        expected = getGaussExpected(nparr, x, y)
    print(expected)
    params,cov=curve_fit(gauss,x,y,expected)
    sigma=sqrt(diag(cov))
    print("Least Squares Sum:", least_squares(params, x, y))
    plot(x,gauss(x,*params),color='red',lw=3,label='model')
    legend()
    print(pd.DataFrame(data={'params':params,'sigma':sigma},index=gauss.__code__.co_varnames[1:]))
    return params
    
    
def bimodalGraph(col, table, expected=None): #change bins?
    '''Bimodal Graph for histogram of col from table with 100 bins.
    Based upon: https://stackoverflow.com/questions/35990467/fit-two-gaussians-to-a-histogram-from-one-set-of-data-python
    '''
    nparr = np.asarray(table[col].tolist())
    y,x,_=hist(nparr,100,alpha=.3,label='data')
    x=(x[1:]+x[:-1])/2
    if expected is None:
        print("getBimodalExpected was called.")
        expected = getBimodalExpected(nparr, x, y)
    print("Expected values: ")
    print(expected)
    params,cov=curve_fit(bimodal,x,y,expected)
    sigma=sqrt(diag(cov))
    print("Least Squares Sum:", least_squares(params, x, y))
    plot(x,bimodal(x,*params),color='red',lw=3,label='model')
    legend()
    print(pd.DataFrame(data={'params':params,'sigma':sigma},index=bimodal.__code__.co_varnames[1:]))
    return params

def getGaussExpected(nparr, x, y):
    maxloc = np.argmax(y)
    maxlocx = x[maxloc]
    ymax = np.max(y)
    return [maxlocx,20,ymax]

def getBimodalExpected(nparr, x, y):
    '''Get's an exected bimodal tuple from the data to be passed into scipy's curve fit fn.
    Procedure: 
        1. Find the peak of zeros (largest peak within the first half of data)
        2. Find the min in the first half (end of first normal, start of second)
        3. Find location matching min in second half (end of second normal)
        4. Find max past 4 times the location of peak of zeros (peak of second normal)
        5. Manipulate and return.
        '''
    #find where max occurs in the first half (0 - 100) (zero's peak, middle of first)
    maxloc = np.argmax(y[0:50])
    maxlocx = x[maxloc]
    ymax = np.max(y[0:50])
    
    #find min in the first half (end of first normal, start of second)
    miny = np.argmin(y[0:50])
    minx = x[miny]
    
    #find matching end location in the second half (end of second)
    endminx = x[50 + find_nearest(nparr[50:100], miny)]
    
    #find max after zero's peak times 4 (middle of second)
    maxlocsecond = maxloc * 4 + np.argmax(y[maxloc * 4 :100])
    ymaxsecond = np.max(y[maxloc * 4:100])
    maxlocsecondx = x[maxlocsecond]
    
    return (maxlocx,20,ymax,maxlocsecondx,30,ymaxsecond)

def find_nearest(array, value):
    '''This function finds the location of the nearest values.'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def least_squares(params, x, y):
    l = 0
    for xcord, ycord in zip(x, y):
        if (len(params) == 3):
            l += (gauss(xcord, params[0], params[1], params[2]) - ycord) ** 2
        else:
            l += (bimodal(xcord, params[0], params[1], params[2], params[3], params[4], params[5]) - ycord) ** 2
    return l