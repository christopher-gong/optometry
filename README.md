# Optometry - Normalization

This repository performs the normalization techniques to prepare for different clustering algorithms to be applied. This project is a UC Berkeley Research Apprenticeship under Amanda McLaughlin in Puthussery & Taylor Labs. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need Python, and a copy of normalization.py in your local directory.

## Example Usage of combine.py

Below is an example on how to use the combine.py methods.

### Import combine and Pandas

These imports are necessary to run the methods.

```
from importHelpers.combine import *
import pandas as pd
```

### combine
This method will normalize and run the PCA n times on the initial data frame passed in, and return a dataframe with the results combined.

```
df = combine(initial, 10)
```

## Example Usage of cluster.py

Below is an example on how to use the cluster.py methods.

### Import cluster and Pandas

These imports are necessary to run the methods.

```
from importHelpers.cluster import *
import pandas as pd
```

### cluster
This method will cluster the initial dataframe passed into nc clusters, with a standard deviation threshold of std_thresh, and return 2 dataframes: one sorted by initial index, and the other sorted by cluster.

```
out_df, o_outdf = cluster(initial, nc = 7, std_thresh = 20, output = False)
```
### Running the Normalization Procedure

## Example Usage of Normalization.py

Below is an example on how to use the normalization.py methods.

### Import Normalization and Pandas

These imports are necessary to run the methods.


import normalization as nrm
import pandas as pd
```

### Running the Normalization Procedure
#### Warning: This may take a long time. Set quiet to False to be able to track progress. 
This Performs the Normalize Procedure on the table. The specific steps performed are explained in the comments of the Normalization file. This method takes the dataFrame and columns that do not want to be normalized. It will perform a standard and z-norm normalization, and return those columns, along with the original columns.

```
newdf = nrm.normalizeProcedure(df, unwantedCols=[], quiet=False)
```
### Histograms
This method should only be run on the dataframe that has applied the above normalization procedure. This method returns three histograms for ease of comparison.

```
nrm.normHists(df, colName)
```

### Mean and Standard Deviation
This method will return a mean and std dataframe for each column in columnsAnalyze, and limit it to first index elements of the dataFrame.

```
mstd = nrm.meanSTD(df, columnsAnalyze, index)
```
## Built With

* [Jupyter Notebook](https://jupyter.org/)

## Authors

* **Christopher Gong** - *Created and tested Normalization File*
