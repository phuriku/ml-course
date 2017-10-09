Mathematics for Machine Learning
================================
There are a number of mathematical pre-requisites for machine learning. Fortunately, these are not as numerous as is often assumed, and much of the mathematics necessary for machine learning is better learned only before the applications for which it is necessary. As such, this section will be concise.

What is Machine Learning?
-------------------------
Most of machine learning can be classified as basic applied statistics. Specifically, machine learning is used to model existing datasets so that predictions and classifications can be made for future data points without manual effort. As such, most of the mathematics behind machine learning involves some degree of linear algebra, probability, and statistics.

Supervised vs. Unsupervised Learning
------------------------------------
To reiterate, machine learning is mostly composed of generating mathematical models for pre-existing datasets so that future data can be classified. In this respect, there are two types of machine learning problems: 1) supervised, wherein a correct classification exists for each point in the existing dataset that can be used to make predictions about incoming data, 2) unsupervised, wherein the existing dataset is raw.

An example of supervised learning would be classifying pictures of dogs and cats using a dataset of 100 pictures for which we have already classified each picture as a dog or a cat. Using this data, a model can be derived that can make predictions of whether a future picture is of a dog or a cat.

An example of unsupervised learning would be clustering points in a dataset. There are no pre-existing correct answers or classifications for the existing data, only raw data points. Using certain algorithms, we can divide data points into natural clusters.

Offline vs. Online Learning
---------------------------
There is also a dichotomy in machine learning between offline and online learning. Offline learning uses a static dataset to generate a model that is fixed even as new data comes in. Online learning produces a dynamic model that is constantly adjusting itself as new data comes in. Online learning tends to be more complex and computationally-intensive than offline learning.

Probability & Statistics
========================
In machine learning, as in the case of almost any modeling based on limited datasets, there is often a large degree of uncertainty. This uncertainty must be quantified, and probability theory is the tool used to do it.

Frequentist vs. Bayesian Statistics
-----------------------------------
There are 2 major ways to think about statistics: Frequentist and Bayesian. Frequentists infer distributions given an existing dataset, while Bayesians additionally rely on a subjective prior hypothesis that guides their assumptions.

For example, in baseball the at-bat success ratio is about 0.26. If a baseball player has 3 hits and 27 outs at bat during his first 10 games, a frequentist will peg the player's future chances of a hit at 0.1. The Bayesian would use the league's average batting average as a prior subjective hypothesis, and naturally modify this subjective hypothesis as more data streams in.

Bayesian statistics is extremely important for our purposes, but will be explained when it becomes necessary. As a brief introduction to Bayesian statistics and the Beta distribution, please refer to [this elegant explanation](http://varianceexplained.org/statistics/beta_distribution_and_baseball/).

Random Variables and Probability Distributions
----------------------------------------------
A *random variable* is a variable that takes on values randomly, based on some probability distribution. A probability distribution is described using a *probability mass function* denoted by **P**. The notation **P**(x = 1) translates to: "the probability that the random variable *x* takes on the value 1."





Linear Algebra
==============
Linear Algebra is a field of mathematics concerned with linear spaces, their elements, and linear transformations of those elements. To be specific, vector spaces are the linear spaces, vectors are the elements, and matrices are the linear transformations of vectors into other vectors.

We will mostly be concerned with matrices and vectors in ℝ<sup>n</sup>.

Vector Spaces
-------------
As mentioned above, a vector space is a linear space composed of mathematical objects called *vectors* that satisfy certain properties. A vector space is defined over a space of *scalars* (usually ℝ or ℂ, the space of real or complex numbers respectively), which can be used to extend vectors in a certain direction.

Formally, a vector space *V* over a scalar space *S* must satisfy the following properties of linearity for all **u**,**v** ∈ *V* and all *c*,*d* ∈ *S*.
1. **u** + **v** is in *V* (closed under addition)
2. **u** + **v** = **v** + **u** (commutative)
3. (**u** + **v**) + **w** = **u** + (**v** + **w**) (associative)
4. There is a **0** vector so that **u** + **0** = **u** (existence of zero vector)
5. There is a **-u** vector so that **u** + (**-u**) = **0**. (existence of additive inverse)
6. *c* · **u** is in *V* (closed under scalar multiplication)
7. *c* · (**u** + **v**) = *c* · **u** + *c* · **v** (distributive property)
8. (*c* + *d*) · **u** = *c* · **u** + *d* · **u** (distributive property)
9. (*c* · *d*) · **u** = *c* · (*d* · **u**) (associative under scalar multiplication)
10. 1 · **u** = **u** (multiplication by scalar 1)

Examples of vector spaces include:
1. ℝ<sup>n</sup> over ℝ
2. ℂ<sup>n</sup> over ℂ
3. ℙ<sub>n</sub> (polynomials of degree <=*n*) over ℝ

*Questions:*
  1. What is the additive inverse of (5,-1) in ℝ<sup>2</sup>?
  2. What is the zero vector in ℙ<sup>n</sup>?

Subspaces
---------

A *subspace* of a vector space *V* is a subset of *V* that also satisfies all of the vector space properties. Since any subset of an existing vector space automatically satisfies all properties except for closure and inclusion of the zero vector, we only need to check properties 1, 4, and 6.

Examples of subspaces:
1. The line *y = x* is a subspace of ℝ<sup>2</sup>
2. ℙ<sub>2</sub> is a subspace of ℙ<sub>3</sub>

Since most of our examples will be in ℝ<sup>n</sup>, this is where we will concentrate our efforts from here on. However, it's important to realize that most of our results can be generalized to any vector space. Vector spaces involving ℂ are also widely applicable in computer science.

*Questions:*
1. Why is the line *y* = *2x+1* not a subspace of ℝ<sup>2</sup>?
2. Why is ℝ<sup>+</sup>, the set of non-negative real numbers, not a subspace of ℝ?

Linear Transformations
----------------------

A *linear transformation* **T** from ℝ<sup>n</sup> → ℝ<sup>m</sup> is a function satisfying the following properties for **u**,**v** ∈ ℝ<sup>n</sup> and *c* ∈ ℝ.
1. **T**(**u** + **v**) = **T**(**u**) + **T**(**v**)
2. **T**(*c* · **u**) = c · **T**(**u**)

In ℝ<sup>n</sup>, matrices *are* the linear transformations. A function is a linear transformation if and only if it can be represented as a matrix. In this schema, elements of ℝ<sup>n</sup> are represented as column vectors in the form:

Then if we have a matrix

If we are working within a fixed vector space ℝ<sup>n</sup>, only square matrices are valid as linear transformations, since an (*m* x *n*)-dimensional matrix multiplied by a (1 x *n*) column matrix yields a (1 x *m*)-dimensional column matrix. (Hence, *m* must equal *n* if we wish to confine ourself to *n*-dimensional space).

Matrix Operations
-----------------

Matrices can be added element-wise:



The *transpose* of matrix

is

Matrices can be multiplied, and the product of matrices **A** and **B** is the linear transformation that results when linear transformation **A** is applied after linear transformation **B** on a vector space. The following matrix properties hold:

1. **A**(**B**+**C**) = **AB** + **AC** (distributive property)
2. **A(BC)** = **(AB)C** (associative property)
3. (**AB**)**<sup>T</sup>** = **B<sup>T</sup>A<sup>T</sup>**
4. There is an identity matrix **I<sub>n</sub>** such that **I<sub>n</sub>u** = **u** for all vectors **u** ∈ ℝ<sup>n</sup>.
