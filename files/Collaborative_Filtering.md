Collaborative Filtering
=======================

The Netflix problem is an interesting one: if you have a large number of users and a large number of items, and each user has rated a small number of items, how can we predict how a user would rate any other item? The problem posed is an example in the field of collaborative filtering, where preferences of a given user can be deduced by observing preferences of similar users.

The Memory-Based Solution
-------------------------
Given a user *u* and a number *N*, we can find the most common users to *u* via normal cosine similarity:<p align="center">
  <img src="https://imgur.com/F1cDaZ7.png" height="48">
</p>

We can then take a weighted average of the ratings of item *i* over each of the top *N* most common users:<p align="center">
  <img src="https://imgur.com/6XNKfd7.png" height="58">
</p>

This is a straightforward approach that's also computationally feasible. Unfortunately, it doesn't work well for sparse ratings matrices (a common scenario in ML, including the Netflix problem), so another solution is needed.

Matrix Factorization
--------------------
For sparse matrices, the Netflix problem is difficult, so let's start with another problem: ordinary least squares. If we have a matrix **A** representing data points and a vector **y** representing the labels, with **x** representing weights, then we have:<p align="center">
  <img src="https://imgur.com/jGMSDlU.png" height="44">
</p>

We have already seen that when λ = 0, this has an exact solution: **x** = (**A<sup>T</sup> A**)<sup>**-1**</sup> **A<sup>T</sup> y**. In fact, there exists a solution for general regularization parameter λ as well: **x** = (**A<sup>T</sup> A** + λ**I<sub>k</sub>**)<sup>**-1**</sup> **A<sup>T</sup> y**.

Unfortunately, as we've already discussed, this exact solution is rarely useful in practice because the matrix **A** could have millions of rows, making matrix inversion an impractical task. Fortunately, there is a case in which this equation can become useful to us.

Let us assume in the Netflix problem that we have *n* users and *m* items. Additionally, each user and each item can be written as a weighted linear combination of *k* common factors. Then the solution is clear: the predicted rating of item *y* by user *x* is just Σ x<sub>i</sub> y<sub>i</sub> ranging from i=1 to *k*. This process is called *matrix factorization*, and this will be our chosen approach. *k* is arbitrary (it's a hidden, implicit variable), but the point is that it should be much smaller than *m* and *n*.

In this case, we can write factorized matrix **X** for users and factorized matrix **Y** for items as follows:<p align="center">
  <img src="https://imgur.com/mQizviX.png" height="78">
</p>

This leads us to the following optimization for **X** and **Y**:<p align="center">
  <img src="https://imgur.com/cTPb6kZ.png" height="68">
</p>

We are minimizing the sum over all observed ratings. However, notice that we are minimizing for **X** and **Y** simultaneously. This makes solving this optimization problem very difficult -- in fact, optimization via traditional methods like stochastic gradient descent are ineffective from a computational standpoint.

However, if we hold one variable fixed at a time while solving for the other variable, and alternating between variables, we can reach a solution. In sudo-code, the approach is as follows:

`while (!converged) {`
<p align="center">
      <img src="https://imgur.com/layKUIq.png" height="38">
</p>
<p align="center">
      <img src="https://imgur.com/UVuEkH1.png" height="38">
</p>
`}`


Note that we have two exact solutions here, one each for **X** and **Y** for as many iterations as it takes to converge. However, **X** and **Y** are both square *k*-dimensional matrices, so as long as *k* is a small number, taking inverses as above won't be computationally-intensive.
