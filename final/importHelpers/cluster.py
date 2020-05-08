import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from importHelpers.response import *
from sklearn.decomposition import PCA

def cluster(initial, std_thresh = 10, nc = 6, clusterin = "cluster_in.csv", clusterout = "cluster_out.csv", output = True):
    initial = frameToSecDF(initial)
    df = removeLowSTD(initial, std_thresh)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(initial)
    principalDf = pd.DataFrame(data = principalComponents)
    pcanc = 3
    while ((sum(principalDf.var()) / sum(initial.var()) < 0.99)):
        #print(pcanc, sum(principalDf.var()) / sum(initial.var()))
        pca = PCA(n_components=pcanc)
        principalComponents = pca.fit_transform(initial)
        principalDf = pd.DataFrame(data = principalComponents)
        pcanc += 1
    #print("PCA NC: ", pcanc)
    kmeans = KMeans(n_clusters=nc).fit(principalDf)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    print("labels: ", labels)
    l = []
    for i in range(nc):
        temp = []
        for j in range(len(labels)):
            if labels[j] == i:
                temp += [j + 1]
        l += [temp]

    out = [0] * 52
    # print("l: ", l)
    dfrows =list(df.index)
    # print("DFR: ", dfrows)
    # print("in: ", list(initial.index))
    # for i in range(len(l)):
    #     for j in l[i + 1]:
    #         print("J: ", j)
    #         print("")
    #         foo = dfrows[j - 1]
    #         out[foo - 1] = i + 1
    # for i in range(len(labels)):
    #     out += [labels[i]]
    out = list(labels)
    # print("out:, ", out)

    data = {'in': list(initial.index), 'out': out}
    outdf = pd.DataFrame.from_dict(data)
    o_outdf = outdf[["out", "in"]].sort_values(by=['out', "in"])

    if output:
        outdf.to_csv(clusterin, index=False, header=False)
        o_outdf.to_csv(clusterout, index=False, header=False)

    return outdf, o_outdf