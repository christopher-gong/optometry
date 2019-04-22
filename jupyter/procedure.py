def ninetyLocator(var):
    for i in range(len(var)):
        if var[i] > 90:
            return i
    return -1

def procedure(roi_cluster_df, condition, outputCSV):
    #use condition
    analysis = []
    if condition == 1:
        comp = 8
        noanalysis = []
    elif condition == 2:
        comp = 8
        noanalysis = ['Islet1']
    elif condition == 3:
        comp = 7
        noanalysis = ['Islet1', 'CD15']
    elif condition == 4:
        comp = 3
        analysis = ['Parv', 'Pax6']
    else:
        raise Exception('condition must be 1 through 4.')

    labelsNOPCA = []
    labelsPCA = []
    
    noanalysis += ['DAPI', 'Tarpg3', 'GLT-1', 'Cav3.1', 'Kv2.2', 'Area', 'Circ','AR','Round','Solidity','stddev', 'median']
    for col in noanalysis:
        roi_cluster_df=roi_cluster_df.drop(roi_cluster_df.columns[roi_cluster_df.columns.str.contains(pat=col)], axis=1)

    if len(analysis) > 0:
        roi_cluster_df = roi_cluster_df[analysis]

    roi_cluster_array = roi_cluster_df.iloc[0:roi_cluster_df.shape[0], 0:roi_cluster_df.shape[1]].values
    roi_cluster_array_noPCA = roi_cluster_array.copy()

    #kmeans and gmm on regular
    kmeans = KMeans(n_clusters=roi_cluster_df.shape[1], random_state = 123) #run on cols
    kmeans.fit(roi_cluster_array_noPCA)

    #roi_clusterResults_df_noPCA=roi_cluster_df.copy()
    #a=kmeans.labels_
    #df_a=pd.DataFrame(a)
    #roi_clusterResults_df_noPCA=pd.concat([roi_clusterResults_df, df_a], axis=1)

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

    #roi_clusterResults_df=roi_cluster_df.copy()
    #a=kmeans.labels_
    #df_a=pd.DataFrame(a)
    #roi_clusterResults_df=pd.concat([roi_clusterResults_df, df_a], axis=1)

    labelsPCA = kmeans.labels_
    #output

    print('PCA: ', labelsPCA)
    print('No PCA: ', labelsNOPCA)