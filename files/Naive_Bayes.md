Naive Bayes
===========

The Naive Bayes method of classification is a method of classifying data into classes given a set of independence assumptions and knowledge of the probability distribution of certain variables.

Bayes Theorem
-------------
Bayes Theorem gives the following formula.

This is due to the fact that,

which follows from the chain rule of probabilities.

This equation is interesting in its own right, but its importance lies in the situation where *p(*y = *y* | x = *x)* is a known distribution while the opposite conditional probability *p(*x = *x* | y = *y)* is more complex. For instance, we can use the fact that *p(*height* = h *| gender* = male)* follows a known probability distribution (the Gaussian with specific mean and variance) to deduce  *p(*gender* = male *| height* = h),* which follows a more complex distribution.

Theory
------

If the random variable *y* determines a probability that the conditional vector *x* should be categorized into class *C<sub>k</sub>*, then this can be written using Bayes Theorem as:

Let us further decompose this into dimensions of the variable *x*:

This is not an ideal form of the equation, but simplifies considerably under the assumption that the dimensions of *x* are independent:

Then the probability that *x = (x<sub>1</sub>, ..., x<sub>n</sub>)* belongs to class *C<sub>k</sub>* is:

However, note that *p(x)* is in both the numerator and denominator of this term and therefore can be excluded from analysis. We therefore have:

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

Given the above table, what is the probability that someone with height 6', weight 130, and foot size 8"
