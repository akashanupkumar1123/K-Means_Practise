K-means Clustering
In this this exercise, you will implement the K-means algorithm and use it for image compression.

You will start with a sample dataset that will help you gain an intuition of how the K-means algorithm works.
After that, you wil use the K-means algorithm for image compression by reducing the number of colors that occur in an image to only those that are most common in that image.



# 1 - Implementing K-means
#
The K-means algorithm is a method to automatically cluster similar
data points together. 

* Concretely, you are given a training set $\{x^{(1)}, ..., x^{(m)}\}$, and you want
to group the data into a few cohesive “clusters”. 


* K-means is an iterative procedure that
     * Starts by guessing the initial centroids, and then 
     * Refines this guess by 
         * Repeatedly assigning examples to their closest centroids, and then 
         * Recomputing the centroids based on the assignments.
         

* In pseudocode, the K-means algorithm is as follows:

    ``` python
    # Initialize centroids
    # K is the number of clusters
    centroids = kMeans_init_centroids(X, K)
    
    for iter in range(iterations):
        # Cluster assignment step: 
        # Assign each data point to the closest centroid. 
        # idx[i] corresponds to the index of the centroid 
        # assigned to example i
        idx = find_closest_centroids(X, centroids)

        # Move centroid step: 
        # Compute means based on centroid assignments
        centroids = compute_means(X, idx, K)
    ```


* The inner-loop of the algorithm repeatedly carries out two steps: 
    * (i) Assigning each training example $x^{(i)}$ to its closest centroid, and
    * (ii) Recomputing the mean of each centroid using the points assigned to it. 
    
    
* The $K$-means algorithm will always converge to some final set of means for the centroids. 

* However, that the converged solution may not always be ideal and depends on the initial setting of the centroids.
    * Therefore, in practice the K-means algorithm is usually run a few times with different random initializations. 
    * One way to choose between these different solutions from different random initializations is to choose the one with the lowest cost function value (distortion).

You will implement the two phases of the K-means algorithm separately
in the next sections. 
* You will start by completing `find_closest_centroid` and then proceed to complete `compute_centroids`.


Finding closest centroids
In the “cluster assignment” phase of the K-means algorithm, the algorithm assigns every training example  𝑥(𝑖)  to its closest centroid, given the current positions of centroids.


Exercise 1
Your task is to complete the code in find_closest_centroids.

This function takes the data matrix X and the locations of all centroids inside centroids
It should output a one-dimensional array idx (which has the same number of elements as X) that holds the index of the closest centroid (a value in  {1,...,𝐾} , where  𝐾  is total number of centroids) to every training example .
Specifically, for every example  𝑥(𝑖)  we set
𝑐(𝑖):=𝑗thatminimizes||𝑥(𝑖)−𝜇𝑗||2,
 
where
𝑐(𝑖)  is the index of the centroid that is closest to  𝑥(𝑖)  (corresponds to idx[i] in the starter code), and
𝜇𝑗  is the position (value) of the  𝑗 ’th centroid. (stored in centroids in the starter code)
If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.





Computing centroid means
Given assignments of every point to a centroid, the second phase of the algorithm recomputes, for each centroid, the mean of the points that were assigned to it.


Exercise 2
Please complete the compute_centroids below to recompute the value for each centroid

Specifically, for every centroid  𝜇𝑘  we set
𝜇𝑘=1|𝐶𝑘|∑𝑖∈𝐶𝑘𝑥(𝑖)
 
where

𝐶𝑘  is the set of examples that are assigned to centroid  𝑘 
|𝐶𝑘|  is the number of examples in the set  𝐶𝑘 
Concretely, if two examples say  𝑥(3)  and  𝑥(5)  are assigned to centroid  𝑘=2 , then you should update  𝜇2=12(𝑥(3)+𝑥(5)) .
If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

2 - K-means on a sample dataset
After you have completed the two functions (find_closest_centroids and compute_centroids) above, the next step is to run the K-means algorithm on a toy 2D dataset to help you understand how K-means works.

We encourage you to take a look at the function (run_kMeans) below to understand how it works.
Notice that the code calls the two functions you implemented in a loop.
When you run the code below, it will produce a visualization that steps through the progress of the algorithm at each iteration.


Random initialization
The initial assignments of centroids for the example dataset was designed so that you will see the same figure as in Figure 1. In practice, a good strategy for initializing the centroids is to select random examples from the training set.

In this part of the exercise, you should understand how the function kMeans_init_centroids is implemented.

The code first randomly shuffles the indices of the examples (using np.random.permutation()).
Then, it selects the first  𝐾  examples based on the random permutation of the indices.
This allows the examples to be selected at random without the risk of selecting the same example twice.

Image compression with K-means
In this exercise, you will apply K-means to image compression.

In a straightforward 24-bit color representation of an image 2 , each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue intensity values. This encoding is often refered to as the RGB encoding.
Our image contains thousands of colors, and in this part of the exercise, you will reduce the number of colors to 16 colors.
By making this reduction, it is possible to represent (compress) the photo in an efficient way.
Specifically, you only need to store the RGB values of the 16 selected colors, and for each pixel in the image you now need to only store the index of the color at that location (where only 4 bits are necessary to represent 16 possibilities).
In this part, you will use the K-means algorithm to select the 16 colors that will be used to represent the compressed image.

Concretely, you will treat every pixel in the original image as a data example and use the K-means algorithm to find the 16 colors that best group (cluster) the pixels in the 3- dimensional RGB space.
Once you have computed the cluster centroids on the image, you will then use the 16 colors to replace the pixels in the original image.


Shape of original_img is: (128, 128, 3)
The code below reshapes the matrix original_img to create an 𝑚×3 matrix of pixel colors (where 𝑚=16384=128×128)


K-Means on image pixels
Now, run the cell below to run K-Means on the pre-processed image.




Compress the image
After finding the top 𝐾=16 colors to represent the image, you can now assign each pixel position to its closest centroid using the find_closest_centroids function.

This allows you to represent the original image using the centroid assignments of each pixel.
Notice that you have significantly reduced the number of bits that are required to describe the image.
The original image required 24 bits for each one of the 128×128 pixel locations, resulting in total size of 128×128×24=393,216 bits.
The new representation requires some overhead storage in form of a dictionary of 16 colors, each of which require 24 bits, but the image itself then only requires 4 bits per pixel location.
The final number of bits used is therefore 16×24+128×128×4=65,920 bits, which corresponds to compressing the original image by about a factor of 6



Finally, you can view the effects of the compression by reconstructing the image based only on the centroid assignments.

Specifically, you can replace each pixel location with the mean of the centroid assigned to it.
Figure 3 shows the reconstruction we obtained. Even though the resulting image retains most of the characteristics of the original, we also see some compression artifacts.



