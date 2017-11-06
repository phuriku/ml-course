Dimensionality Reduction
========================

Dimensionality reduction is one of the most important topics in machine learning, primarily due to its applicability in practice. To summarize, dimensionality reduction is the process of reducing a feature space of dimension *n* to a small feature space of dimension *k* by performing specific transformations and cutting out features that prove insignificant to our model.

(In technical terms, cutting out features is called *feature elimination*. Re-writing our feature space by posing each feature as a linear combination of other features with the end goal of reducing dimensions is called *feature extraction*.)

Perhaps unfortunately, dimensionality reduction requires a deeper knowledge of some aspects of linear algebra, in particular the subtopic of eigenvectors and eigenvalues.

Eigenvectors and Eigenvalues
----------------------------
Recall that an (*m* x *n*)-dimensional matrix **A** represents a linear transformation from ℝ<sup>n</sup> to ℝ<sup>m</sup>. While linear transformations are simple in light of their properties, they are still somewhat abstract. However, we can make them more concrete by analyzing certain properties of linear transformations. Eigenvectors and eigenvalues provide *half* of the solution to this puzzle.

To step back into the abstract world of mathematics, an eigenvector **v** and an eigenvalue *λ* for a matrix **A** are a vector/scalar couple that satisfy the following properties: **A**・**v** = *λ*・**v**. In other words, **A** acts on the eigenvector **v** by simply scaling it by a factor of *λ*.

Eigenvectors and eigenvalues seem very abstract, but in intuitive terms, they represent the notion of scaling in fixed directions. This fact is represented by the the following theorem:

**Matrix Eigendecomposition Theorem**: If **A** is a square (*n* x *n*)-dimensional matrix with *n* linearly independent eigenvectors, then we can write **A** = **Q**・**Λ**・**Q<sup>-1</sup>**, where **Q**'s columns are the eigenvectors of **A** and **Λ** is the diagonal matrix of the eigenvalues of **A**.

(Technical note: This is also sometimes called *spectral decomposition*, especially in pure mathematics.)

To express this in more intuitive terms, **A** can represented as a combination of each of its eigenvectors being scaled by its paired eigenvalue.

Principal Component Analysis (PCA)
----------------------------------

Principal Component Analysis (PCA) uses the Matrix Eigendecomposition Theorem above to perform dimensionality reduction on the feature space. In particular, it treats the covariance matrix **A**・**A<sup>T</sup>** (which features variance of **A** along the diagonal and covariance between respective dimensions elsewhere). Using the fact that **A**・**A<sup>T</sup>** is a square, symmetric matrix, we can use matrix eigendecomposition to write **A**・**A<sup>T</sup>** = **Q**・**Λ**・**Q<sup>-1</sup>**, where **Q** represents the eigenvectors and **Λ** has eigenvectors along the diagonal.

In order to perform feature extraction and reduce the dimensions of the feature space, we want to find the directions of greatest variance. This is done by finding the largest *k* eigenvalues (where we choose *k*), and taking the associated eigenvectors as the new features.

The eigenvector associated with largest eigenvalue (i.e., the direction of the largest scaling factor of the covariance matrix) is called the *first principal component*. Similarly, the eigenvector associated with the 2nd largest eigenvalue is called the *second principal component*, and so on.

<p align="center">
  <img src="https://imgur.com/lU5gXML.png" height="300">
</p>


Singular Value Decomposition
----------------------------
While PCA is a fine method, there is a slight problem: working directly with **A**・**A<sup>T</sup>** can be troublesome to compute when small values are involved. This can be alleviated by using a technique called Singular Value Decomposition.

As we will see in the Singular Value Decomposition (SVD) theorem below, "well-behaved" matrices can be represented as the product of 1 scaling matrix and 2 rotation matrices.

**SVD Theorem**: If **A** is an (*m* x *n*)-dimensional matrix, then **A** can be written as: **A** = **U**・**Λ**・**V<sup>T</sub>**, where **U** and **V** are orthogonal matrices of dimension (*m* x *m*) and (*n* x *n*) respectively, and **Λ** is a diagonal (*m* x *n*)-dimensional matrix.

(Note: An orthogonal matrix is a matrix **A** satisfying **A**・**A<sup>T</sup>** = **I**. Orthogonal matrices represent rotations and reflections. For example, the matrix<p align="center">
  <img src="https://imgur.com/WamXjci.png" height="50">
</p>
represents a rotation by angle θ.)

What the SVD Theorem says in intuitive terms is, *any* matrix **A** can be represented as a rotation or reflection **V<sup>T</sub>**, combined with a scaling diagonal matrix **Λ** and a rotation/reflection **U**.

Since **Λ** is a scaling matrix, we again take the largest *k* values and the rotations/reflections associated with those *k* dimensions as represented within **U** and **V**. This is the dimensionality reduction we were seeking.

A Comparison of PCA and SVD
---------------------------
The final results of PCA and SVD are identical. In fact, it can be shown (almost trivially) that the values of the scaling matrix **Λ** in SVD are the eigenvalues of **A**・**A<sup>T</sup>**.

However, there is a subtle difference: in PCA, we deal directly with the matrix **A**・**A<sup>T</sup>**, whereas we do not consider it explicitly in SVD. As mentioned before, this has serious repercussions for PCA when **A** contains entries that are small, or even if **A** is sparse.

SVD has its own sets of techniques for finding the associated orthogonal and scaling matrices. These are well-developed and often algorithmically faster than the techniques for PCA.

From an academic viewpoint, PCA seems like a bit of a hack. This is because we're working with an (**m** x **n**)-dimensional matrix, and eigendecomposition only works for square matrices. Hence we multiply by the transpose to get the covariance matrix, which can be a costly operation when **A** is large. As we've mentioned, this also isn't a trustworthy approach when the entries of **A** are small. Personally, I've found that PCA is good for finding an intuition, while SVD is better in practice.

Interestingly enough, we can use SVD to create a safer PCA algorithm: We first write **A** = **U**・**Λ**・**V<sup>T</sub>** by the SVD Theorem, then for covariant matrix **C**, we write **C** = **A**・**A<sup>T</sup>** = (**U**・**Λ**・**V<sup>T</sub>**)・(**U**・**Λ**・**V<sup>T</sub>**)<sup>**T**</sup> = (**U**・**Λ**・**V<sup>T</sub>**)・(**V**・**Λ**・**U<sup>T</sup>**) = (**U**・**Λ**<sup>**2**</sup>・**U<sup>T</sup>**). This, of course, relies on already knowing the SVD decomposition, making it only a theoretically interesting result.
