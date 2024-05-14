The task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters).
Examples:

Natural Language Processing (NLP)

Computer Vision

Stock markets

Customer / Market Segmentation

Types:
Connectivity-based clustering
Distance based
E.g., Hierarchical clustering
Centroid-based clustering
Represents each cluster by a single mean vector
E.g., k-means algoritm
Distribution-based clustering
Modeled using statistical distributions
E.g., Multivariate normal distributions used by the expectation-maximization algorithm.
Density-based clustering
Defines clusters as connected dense regions in the data space.
E.g., DBSCAN





Ward’s Agglomerative Hierarchical Clustering
Wikipedia

Agglomerative:
Bottom up
Each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
Divisive:
Top down
All observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.
Stackexchange

CMU Notes

PSE Stat505 Linkage Methods:

Single Linkage: shortest distance. Distance between two clusters to be the minimum distance between any single data point in the first cluster and any single data point in the second cluster.

Complete Linkage: Furthest distance. Distance between two clusters to be the maximum distance between any single data point in the first cluster and any single data point in the second cluster.

Average Linkage: Average.

Centroid Method: Distance between two clusters is the distance between the two mean vectors of the clusters.

Ward’s Method: ANOVA based approach.

Iterative process
Minimises the total within cluster variance
At each step, the pair of clusters with minimum between cluster distance are merged






k-Means Clustering
Analyse and find patterns / clusters within data

Distance measures

scikit learn

Clusters data by trying to separate samples in n groups of equal variance
Minimizing a criterion known as the inertia or within-cluster sum-of-squares.
Requires the number of clusters to be specified.
Scales well
How does it work?

Divides a set of samples into disjoint clusters
Each described by the mean of the samples in the cluster.
The means are commonly called the cluster “centroids”
Note that the centroids are not, in general, points from, although they live in the same space.
The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum of squared criterion





Some Challenges:

The globally optimal result may not be achieved

The number of clusters must be selected beforehand

k-means is limited to linear cluster boundaries

k-means can be slow for large numbers of samples

Elbow Method
Use intrinsic metrics
An example fo this is the within-cluster Sums of Squared Error
scikit learn has already provided it via inertia_ attribute





8.0 Silhouette Analysis
silhouette score=𝑝−𝑞𝑚𝑎𝑥(𝑝,𝑞)
 
𝑝  is the mean distance to the points in the nearest cluster that the data point is not a part of

𝑞  is the mean intra-cluster distance to all the points in its own cluster.

The value of the silhouette score range lies between -1 to 1.

A score closer to 1 indicates that the data point is very similar to other data points in the cluster,

A score closer to -1 indicates that the data point is not similar to the data points in its cluster.





Mean Shift
wikipedia

Non-parametric
Identify centroids location

For each data point, it identifies a window around it
Computes centroid
Updates centroid location
Continue to update windows
Keep shifting the centroids, means, towards the peaks of each cluster. Hence the term Means Shift
Continues until centroids no longer move
Used for object tracking



