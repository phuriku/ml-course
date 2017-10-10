Mathematics for Machine Learning
================================
There are a few mathematical pre-requisites for machine learning, including elementary probability theory and linear algebra. Fortunately, much of the mathematics necessary for ML is better learned directly before the applications for which it is necessary. This section will only cover the content necessary to understand even the most elementary parts of machine learning.

What is Machine Learning?
-------------------------
Most of machine learning can be classified as applied statistics. Specifically, machine learning consists of modeling existing datasets so that predictions can be made for future data without human intervention. Consequently, most of the mathematics behind machine learning involves some degree of linear algebra, probability, and statistics.

Supervised vs. Unsupervised Learning
------------------------------------
Machine learning mostly consists of generating mathematical models for pre-existing datasets so that future data can be classified. In this respect, there are two types of machine learning problems: 1) *supervised learning*, where a correct label is known and assigned to each point in an existing dataset, and 2) *unsupervised learning*, where the existing dataset is raw in the purest sense.

An example of supervised learning is classifying pictures of dogs and cats using a dataset of 100 pictures for which we have already manually classified each picture as a dog or a cat. Using this pre-labeled data, a model can be derived to make predictions about whether other pictures are of dogs or of cats.

An example of unsupervised learning is clustering points in a dataset. There are no classifications for the existing data, only raw data points. Using mathematical algorithms, we can separate data points into natural clusters.

Offline vs. Online Learning
---------------------------
There is also a dichotomy in ML between offline and online learning.

*Offline learning* uses a *static* dataset to generate a model that remains unmodified even as new data comes in. On the other hand, *online learning* produces a dynamic model that constantly adjusts itself with new data points. Online learning tends to be more complex and computation-intensive than offline learning.

Probability & Statistics
========================
As in the case of almost any modeling based on limited datasets, there is often a large degree of uncertainty in ML. This uncertainty must be quantified, and probability theory is the tool used to do so.

Frequentist vs. Bayesian Statistics
-----------------------------------
There are 2 major ways to think about statistics: Frequentist and Bayesian. Frequentists infer distributions based only on the existing data. In contrast, Bayesians rely on a prior hypothesis that guides their initial assumptions and slowly change their view as new data comes in.

In major league baseball, the median at-bat success ratio is about 0.26. If a baseball player has 3 hits and 27 outs at bat during his first 10 games, a frequentist will peg the player's future expected chances of a hit at 0.1 with a large potential variance. The Bayesian would use the league's median batting average as a prior hypothesis, and naturally modify this view as more data becomes available.

Bayesian statistics is extremely important for our purposes, but will be explained as it becomes relevant. As a brief introduction to Bayesian statistics and the Beta distribution, please refer to [this elegant explanation](http://varianceexplained.org/statistics/beta_distribution_and_baseball/) that fleshes out the example above.

Random Variables and Probability Distributions
----------------------------------------------
A *random variable* is a variable that takes on values randomly, based on some probability distribution. A probability distribution is described using a *probability mass function* denoted by **P**. The notation **P**(x = 1) translates to: "the probability that the random variable *x* takes the value 1."

A probability distribution can be either *continuous* or *discrete*. Discrete distributions take a finite (or countably-infinite) number of values, while continuous distributions take values along a continuous spectrum (such as ℝ).

For example, the uniform distribution (where *n* values each occur at a probability of 1/*n*) is discrete. A fair dice roll is an example of a uniform distribution with *n*=6. The normal distribution, covered later, is a continuous distribution.

In a continuous distribution, probabilities are not taken over points, but over *ranges*. For example, the probability that a random variable *x* takes values between 1 and 2 is described by ∫<sub>1</sub><sup>2</sup>**P**(x).

In the discrete case, probabilities of individual values must sum to 1: Σ<sub>x</sub>**P**(x) = 1. In the continuous case, this is an integral: ∫**P**(x) = 1. This implies that for discrete distributions, the probability of an individual value cannot exceed 1, whereas for continuous distributions, the probability of the value lying in a specified *range* cannot exceed 1.

Joint Probability Distributions and Independence
------------------------------------------------

Probability distributions can model many variables at the same time. A *joint distribution* **P**(*x* = 1, *y* = 0) denotes the joint probability that the random variable *x* is 1 and *y* is 0 simultaneously.

We say that two random variables *x* and *y* are *independent* if their joint probability distribution is the product of the two distributions taken independently:<p align="center">
  <img src="https://i.imgur.com/qCE0UpX.png" height="24">
</p>

For example, if a random variable *x* models the toss of a 6-sided die, while *y* models the flip of a coin, then these are clearly independent, *e.g.* <p align="center">
  <img src="https://i.imgur.com/mJSGPG2.png" height="26">
</p>

However, if we have a standard deck of 52 cards and *x* and *y* are random variables representing a draw from the same deck, then these are not independent distributions, because the probability of both drawing an ace is <p align="center">
  <img src="https://i.imgur.com/1S2Rd1a.png" height="27">
</p>, whereas <p align="center">
  <img src="https://i.imgur.com/AMGIibd.png" height="26">
</p>

Conditional Probability
-----------------------

Probabilities can also be calculated on the condition that another event happens. The probability that the random variable *x* = x *given* that *y* = y can be calculated using the formula:<p align="center">
  <img src="https://i.imgur.com/DDIa5j9.png" height="32">
</p>

*Questions:*
- This formula isn't useful when *x* and *y* are independent. Why?

Expectation, Variance, and Covariance
-------------------------------------
The *expectation* or *expected value* of a random variable is the mean value taken by its probability distribution. In symbols, the expected value of *x* is written 𝔼[x] = Σ<sub>x</sub>xP(x) for discrete distributions, and 𝔼[x] = ∫xP(x) for continuous distributions.

The *standard deviation* of a random variable *x* describes how widely the values of *x* vary across the distribution. It is described by the average deviance from the mean: SD(x) = 𝔼[|x - 𝔼(x)|]. The *variance* Var(x) is the standard deviation squared, and is more commonly used than SD(x).

The *covariance* between two random variables *x* and *y* describes how they vary in relation to each other: Cov(x,y) = 𝔼[(x - 𝔼(x))(y - 𝔼(y))]. The covariance is positive for variables that vary similarly, negative for variables that vary inversely, and approximately 0 for variables with little similarity.

Common Probability Distributions
--------------------------------
The *Bernoulli distribution* is a distribution controlled by a single parameter *p* between 0 and 1, which represents the chances of a success. It has the following properties:
1. **P**(x = 1) = *p*
2. **P**(x = 0) = 1 - *p*
3. 𝔼[x] = *p*

The *Binomial distribution* is the Bernoulli distribution extended over *n* turns. It represents the probability of *m* successes over *n* turns, where each turn has a success probability of *p*. It has the following properties:
1. **P**(x = *n*) = *p*<sup>*n*</sup>
2. **P**(x = 0) = (1-*p*)<sup>*n*</sup>
3. **P**(x = *m*) = (*n*,*m*) *p*<sup>*m*</sup>(1-*p*)<sup>*n*-*m*</sup>
3. 𝔼[x] = nφ

The following image is the binomial distribution for *p* = 0.5, *n* = 15:<p align="center">
  <img src="https://i.imgur.com/przhOUO.png" height="180">
</p>

The *Gaussian distribution* or *normal distribution* is the most commonly-known distribution, primarily because it has a tendency to describe naturally-occurring distributions (*c.f.* the Central Limit Theorem). It is described by mean *μ* and variance *σ*<sup>2</sup>: <p align="center">
  <img src="https://i.imgur.com/lMpXZX7.png" height="70">
</p>

For the values of *μ* and *σ* in the legend, the Gaussian distribution looks like: <p align="center">
  <img src="https://i.imgur.com/53KA7PA.png" height="200">
</p>

The Gaussian distribution can be used to approximate the binomial distribution with parameters *μ* = *np* and *σ*<sup>2</sup> = *np*(1-*p*).

The *Beta distribution* is one of the most useful distributions in statistics, particularly in the Bayesian model. Unfortunately, it has a complex form and is rather computation-intensive, often making it unfeasible from a pragmatic ML perspective. Many of its practical usages can be approximated by a Gaussian distribution.

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

In ℝ<sup>n</sup>, matrices *are* the linear transformations. A function is a linear transformation if and only if it can be represented as a matrix. In this schema, elements of ℝ<sup>n</sup> are represented as (1 x *n*)-dimensional column vectors in the form:<p align="center">
  <img src="https://i.imgur.com/dw9cKBO.png" height="100">
</p>

Then if we have a matrix **A** represented by:<p align="center">
  <img src="https://imgur.com/3N0XAVx.png" height="100">
</p>

then matrix **A** applied to the vector **u** yields:<p align="center">
  <img src="https://i.imgur.com/aWhbl4u.png" height="100">
</p>

In words, matrices transform column vectors into column vectors, and this is the intuition behind their existence as linear operators on ℝ<sup>n</sup>.

If we are working within a fixed vector space ℝ<sup>n</sup>, only square matrices are valid as linear transformations, since an (*m* x *n*)-dimensional matrix multiplied by a (1 x *n*) column matrix yields a (1 x *m*)-dimensional column matrix. (Hence, *m* must equal *n* if we wish to confine ourself to *n*-dimensional space).

Matrix Operations
-----------------

Matrices can be added element-wise:<p align="center">
  <img src="https://i.imgur.com/5zmKJpi.png" height="60">
</p>

The *transpose* of matrix **A** is denoted **A<sup>T</sup>**, and it inverts along the diagonal:<p align="center">
  <img src="https://i.imgur.com/xM1rLLp.png" height="130">
</p>

Matrices can be multiplied, and the product of matrices **A** and **B** is the linear transformation that results when linear transformations **A** and **B** are applied to a vector consecutively. For an (*n* x *m*)-dimensional matrix **A** and an (*m* x *p*)-dimensional matrix **B**, the formula is:<p align="center">
  <img src="https://i.imgur.com/QwaOvWy.png" height="60">
</p>

 The following matrix properties hold:

1. **A** (**B**+**C**) = **A B** + **A C** (distributive property)
2. **A** (**B C**) = (**A B**) **C** (associative property)
3. (**A B**)**<sup>T</sup>** = **B<sup>T</sup> A<sup>T</sup>**
4. There is an identity matrix **I<sub>n</sub>** such that **I<sub>n</sub>u** = **u** for all vectors **u** ∈ ℝ<sup>n</sup>:<p align="center">
  <img src="https://i.imgur.com/yhobstx.png" height="130">
</p>

An (*n* x *n*)-dimensional matrix **A** is said to have an inverse if there is a matrix **A<sup>-1</sup>** such that **A** **A<sup>-1</sup>** = **A<sup>-1</sup>** **A** = **I<sub>n</sub>**. Methods for computing an inverse are complex, but many computationally-efficient techniques exist. A matrix has an inverse only if its determinant is non-zero. (The *determinant* of a matrix is a real number that can be viewed as the scaling factor of the linear transformation corresponding to the matrix. It is a complex topic too involved for this brief introduction.)
