README: Randomized SVD Analysis and Data Compression Applications
Objective
This notebook explores and illustrates the Randomized Singular Value Decomposition (SVD) technique, also known as Truncated SVD, with a particular focus on its application in compressing large datasets. We will examine how SVD can be used to reduce data dimensionality while retaining most of the important information.

Theoretical Basis
1. Singular Value Decomposition (SVD)
SVD is a powerful matrix factorization method that decomposes a matrix  A  into three matrices:  A=UΣVT . The singular values  σi  in the  Σ  matrix indicate the importance of each component in reconstructing the original matrix.

2. Compact SVD (Truncated SVD)
When we are only interested in the principal contributing components, we use Compact SVD, retaining only the  r  largest singular values and their corresponding singular vectors. This helps reduce matrix size and eliminate noise.

3. Randomized SVD
For very large matrices, Randomized SVD is an efficient method to approximate the largest singular values and singular vectors. It uses random projections to reduce the matrix's dimensionality, then performs SVD on this smaller matrix to speed up computation.

4. Significance of Data Energy and Criterion for Choosing k
"Data energy" is measured by the sum of the squares of the singular values (Frobenius norm). For data compression, we select a rank  k  such that the sum of the squares of the  k  largest singular values reaches a certain proportion  η  of the total original energy, for example,  90% ,  95% , or  99% .

Data Source
We use the MovieLens 25M Dataset from GroupLens Research. This dataset contains 25 million movie ratings, including:

ratings.csv: Contains user IDs, movie IDs, and ratings.
tags.csv: Contains user comments.
movies.csv: Contains movie titles and genres.
We will focus on ratings.csv to build a user-movie rating matrix.

Implementation Steps
Data Preparation: Download and extract the MovieLens 25M dataset. Read ratings.csv, tags.csv, and movies.csv files into Pandas DataFrames.
Data Preprocessing: Construct a sparse user-movie matrix (rating_matrix) from ratings.csv, where rows are users, columns are movies, and values are ratings.
Perform Randomized SVD: Use the randomized_svd function from scikit-learn to compute  U ,  Σ  (singular values), and  VT  for the rating_matrix. To ensure efficiency, we compute a limited number of the largest singular values (e.g., 500).
Determine k based on Energy Ratio: Develop a function to select the optimal number of components  k  based on a desired energy retention threshold (e.g., 90%, 95%, 99%).
Construct Low-Rank Approximation Matrix  Ak : Extract  Uk ,  Σk ,  VTk  with the chosen  k . Note that reconstructing the full  Ak  as a dense matrix can be memory-intensive.
Evaluate Error: Calculate the approximation error using the Frobenius norm ( ∥A−Ak∥F ), based on the singular values not retained.
Compare Storage Capacity: Compare the number of elements required to store the original sparse matrix and the compressed matrices ( Uk,Σk,VTk ).
Key Results
Original Matrix Dimensions:  162541×59047  with  25,000,095  non-zero elements.
Chosen k: For a desired energy retention of  95% , we selected  k=392 .
Retained Energy:  208,239,837.66  (equivalent to  95.03%  of total energy).
Frobenius Error:  3298.87  with  k=392 .
Storage Capacity:
Number of non-zero elements in the original matrix:  25,000,095 .
Number of elements in the compressed representation ( Uk,Σk,VTk ):  86,862,888 .
In this case, because the original matrix is very sparse and  k  is relatively large, the dense compressed representation consumes more storage than storing the original sparse matrix (Estimated storage reduction:  −247.45% ).
Conclusion
Randomized SVD is a powerful tool for dimensionality reduction in large and sparse matrices. Although storing  Uk,Σk,VTk  directly can be more costly than the sparse representation of the original matrix when the matrix is very sparse, the true value of Randomized SVD lies in its ability to extract principal components and enable efficient work with a low-rank approximation of the data. This is highly valuable in applications such as recommender systems, principal component analysis, and image processing, where dimensionality reduction is key to improving performance and scalability.

