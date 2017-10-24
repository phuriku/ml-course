Gradient Descent
----------------

Gradient descent is one of the most commonly used numerical techniques for optimization. In short, it uses the fact that the steepest descent is in the opposite direction of the gradient vector to numerically approximate a minimum. By taking strategically-sized steps in the direction of steepest descent, a local minimum can eventually be found without difficulty.

In mathematical terms, the gradient descent method performs the following iterations, depending on step size η:<p align="center">
  <img src="https://imgur.com/XovaZyt.png" height="38">
</p>
, where L(w) signifies the loss function as applied to the predictor function with weights w.

Choosing the correct value of η is of great importance. If it is too big, it will overshoot the minimum, while if it is too small, it is a waste of computational resources.

Remember that our purpose is to minimize the loss function: hence, this will be the function upon which we apply gradient descent. From a computational perspective, taking the gradient (or regular derivative) of a function is difficult -- for this reason, packages like Spark MLLib only provide the gradient for a limited set of loss functions.

Stochastic Gradient Descent
---------------------------
Even when the gradient of the loss function is known, gradient descent can become impractical when we have a very large dataset. (Recall that loss functions like MSE require summing over all data points.) We can avoid this problem with the use of *mini-batching*, which in essence is random sampling of the large dataset. A surprisingly accurate approach, mini-batching allows us to work with datasets with billions of data points without encountering performance difficulties.
