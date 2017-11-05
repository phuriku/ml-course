Naive Bayes
===========

The Naive Bayes method of classification is a method of classifying data into classes given a set of independence assumptions of the dependent variables.

Bayes Theorem
-------------
Bayes Theorem gives the following formula:<p align="center">
  <img src="https://imgur.com/S3SqTaw.png" height="62">
</p>

This is due to the fact that,<p align="center">
  <img src="https://imgur.com/Z1q0srb.png" height="52">
</p>

which follows from the chain rule of probabilities.

This equation is interesting in its own right, but its importance lies in the situation where *P(*y = *y* | x = *x)* is a known distribution while the opposite conditional probability *P(*x = *x* | y = *y)* is more complex. For instance, we can deduce  *p(*gender* = male *| height* = h),* which follows a complex distribution, use the fact that *p(*height* = h *| gender* = male)* follows a known probability distribution (the Gaussian with specific mean and variance).

Theory
------

If the random variable *y* determines a probability that the conditional vector *x* should be categorized into class *C<sub>k</sub>*, then this can be written using Bayes Theorem as:<p align="center">
  <img src="https://imgur.com/ho5S410.png" height="58">
</p>

Let us further decompose this into dimensions of the variable *x*:<p align="center">
  <img src="https://imgur.com/vjynMPW.png" height="64">
</p>

This is not an ideal form of the equation, but simplifies considerably under the assumption that the dimensions of *x* are independent:<p align="center">
  <img src="https://imgur.com/Oe1pz2i.png" height="32">
</p>

Then the probability that *x = (x<sub>1</sub>, ..., x<sub>n</sub>)* belongs to class *C<sub>k</sub>* is:<p align="center">
  <img src="https://imgur.com/Fqkhs0Q.png" height="58">
</p>

However, note that *P(x)* is in both the numerator and denominator of this term and therefore can be excluded from analysis. We therefore have:<p align="center">
  <img src="https://imgur.com/8hEOSxj.png" height="64">
</p>

Although this equation seems complicated, it's not. We've removed the *P(x)* term, which is intuitively difficult to reason about. More importantly, the denominator is just the sum of the numerator across all classes.

An Example
----------

Assume we have the following dataset:

| Gender | Height	      | Weight | Foot Size
|:------:|:------------:|:------:|:--------:
| male   | 6	          | 180lb  | 12"
| male	 | 5.92 (5'11") |	190lb  | 11"
| male	 | 5.58 (5'7")  |	170lb	 | 12"
| male	 | 5.92 (5'11") |	165lb	 | 10"
| female | 5	          | 100lb	 | 6"
| female | 5.5 (5'6")	  | 150lb	 | 8"
| female | 5.42 (5'5")	| 130lb	 | 7"
| female | 5.75 (5'9")	| 150lb	 | 9"

Given the above table, what is the probability that someone with height 6', weight 130 lb., and foot size 8" is a male?

Let us first note that we have two classes here: male and female. We can further assume that the population is 50% male and 50% female.

Now we have to calculate the three values *pdf(height | male)*, *pdf(weight | male)*, and *pdf(foot size | male)*. Assuming that all of these are Gaussian distributions, we can calculate the mean and variance of the distributions by averaging across the male data points and finding the average square distance from the mean, respectively. This yields the following values for males:


|                   | Height | Weight   | Foot Size
|:-----------------:|:------:|:--------:|:--------:
| **μ**             | 5.855  | 176.25lb | 11.25"
| **σ<sup>2</sup>**	| 0.026  |	92.2lb  | 0.6875"

Now in order for us to find *pdf(*height* = 6' | *male*),* we simply plug height = 6' into the Gaussian distribution with mean μ=5.855 and σ<sup>2</sup>=0.026, and get *pdf(*height = *6'* | male*)* = 1.6496. Similarly, we get *pdf(*weight = *130lb* | male*)* = 0.00000038 and *pdf(*foot size = *8"* | male*)* = 0.00022185. This means that the numerator of our formula for males is 0.5 x 1.6496 x 0.00000038 x 0.00022185 = 6.953E-11.

Similarly, we calculate the numerator for females to be 0.5 \* 0.14422184 \* 0.01935055 \* 0.32286937 = 4.505E-4. Adding the figures for males and females yields the denominator, and we can see the probability that the person at hand is a male is 1.5E-7, or 0.000015%.
