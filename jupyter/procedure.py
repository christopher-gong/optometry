import normalization as nrm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def ninetyLocator(var):
    #locates index over 90
    for i in range(len(var)):
        if var[i] > 90:
            return i
    return -1

def colorCSV(labels, colors, csvLoc):
    #outputs the color CSV to the location.
    results = []
    for c in labels:
        results += [colors[int(c)]]
    df = pd.DataFrame(data={'COLORS': results})
    df.to_csv(csvLoc, index=False, header=False)
    return df

def procedure(roi_cluster_df, dropkeep, dkcols, comp, outputCSV, colors=['#ff0000', '#ffa500', '#ffff00', '#00ff00', '#0000ff', '#551a8b', '#ffc0cb', '#8b4513', '#d3d3d3', '#000000', 'uh', 'oh', 'rip']):
    #drop is true, keep is false
    assert dropkeep == 'drop' or dropkeep == 'keep', "dropkeep must be string drop or keep"

    roi_cluster_df = roi_cluster_df.copy() #creates new df
    colNames = roi_cluster_df.columns[roi_cluster_df.columns.str.contains(pat='z norm')] #keeps only znorm data
    roi_cluster_df = roi_cluster_df[colNames]
    roi_cluster_df = roi_cluster_df.drop(['z norm total cell index', 'z norm ind cell index'], axis=1)

    labelsNOPCA = []
    labelsPCA = []

    #use condition and drop/keep cols
    if dropkeep == 'keep':
        z = []
        for col in list(roi_cluster_df):
            for a in dkcols:
                if a in col:
                    z += [col]
        roi_cluster_df = roi_cluster_df[z]
    else:
        dkcols += ['DAPI', 'Tarpg3', 'GLT-1', 'Cav3.1', 'Kv2.2', 'Area', 'Circ','AR','Round','Solidity','stddev', 'median']
        for col in dkcols:
            roi_cluster_df = roi_cluster_df.drop(roi_cluster_df.columns[roi_cluster_df.columns.str.contains(pat=col)], axis=1)

    roi_cluster_array = roi_cluster_df.iloc[0:roi_cluster_df.shape[0], 0:roi_cluster_df.shape[1]].values
    roi_cluster_array_noPCA = roi_cluster_array.copy()

    #kmeans and gmm on regular
    kmeans = KMeans(n_clusters=roi_cluster_df.shape[1], random_state = 123) #run on cols
    kmeans.fit(roi_cluster_array_noPCA)
    gmm = GaussianMixture(n_components=comp).fit(roi_cluster_array_noPCA)

    gmm_labelsNOPCA = gmm.predict(roi_cluster_array_noPCA)
    labelsNOPCA = kmeans.labels_

    #run pca
    covar_matrix=PCA(n_components=comp) #its comp right
    covar_matrix.fit(roi_cluster_df)
    variance=covar_matrix.explained_variance_ratio_
    var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
    ncomp = ninetyLocator(var)

    pca_num=PCA(n_components=comp)
    pca_array=pca_num.fit_transform(roi_cluster_df)

    #kmeans and gmm on pca
    kmeans = KMeans(n_clusters=ncomp, random_state = 123) #make sure this is correct. clusters vs comp
    kmeans.fit(pca_array)
    gmm = GaussianMixture(n_components=comp).fit(pca_array)

    gmm_labelsPCA = gmm.predict(pca_array)
    labelsPCA = kmeans.labels_
    
    #output
    print('K means PCA: ', labelsPCA)
    print('K means No PCA: ', labelsNOPCA)
    print('GMM: ', gmm_labelsPCA)
    print('GMM No PCA: ', gmm_labelsNOPCA)

    colorCSV(labelsPCA, colors, outputCSV + "K_MEANS_PCA.csv")
    colorCSV(labelsNOPCA, colors, outputCSV + "K_MEANS_NOPCA.csv")
    colorCSV(gmm_labelsPCA, colors, outputCSV + "GMM_PCA.csv")
    colorCSV(gmm_labelsNOPCA, colors, outputCSV + "GMM_NOPCA.csv")
