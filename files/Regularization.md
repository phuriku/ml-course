Regularization
--------------

We previously alluded to the idea that overfitting is a real problem in machine learning, and that we should keep models simple for the purposes of generalization. In previous examples, this involved limiting the *hypothesis space* - the space of all valid models - to linear functions.

While this is a valid approach, it is arbitrary and inexact. Why can we use functions like *y = 0.987x + 3.215* but we can't use *y = x<sup>2</sup>*? There is a better approach, and this more generalized approach of restricting the hypothesis space in a more *continuous* fashion is called *regularization*.

We can add a regularization term to our original loss function to add a penalty to certain weights. For example, the popular L<sup>2</sup> regularization term added to the least squares error (LSE) loss function results in the following *generalized loss function* J(w). The λ is called the *regularization parameter*, and can be set to any number >0. Common values of λ in practice are between 0 and 0.5.<p align="center">
  <img src="https://imgur.com/7TgwpYc.png" height="52">
</p>

In L<sub>2</sub> regularization, we are penalizing larger values of w<sub>i</sub> for the purposes of minimizing complexity. L<sub>1</sub> regularization is similar:<p align="center">
  <img src="https://i.imgur.com/J5WND4i.png" height="52">
</p>

There is a 3rd type of regularization called *elastic net regularization* that combines L<sub>1</sub> and L<sub>2</sub> regularization:<p align="center">
  <img src="https://imgur.com/nyV6Vh8.png" height="58">
</p>

The following terminology is also noteworthy: Linear least squares together with L<sub>2</sub> regularization is often called *ridge regression* in the context of statistics, while linear least squares with L<sub>1</sub> regularization is often called *LASSO regression*.

Which Regularization To Use?
----------------------------

The question then becomes, which type of regularization is better? The answer is, of course, that it largely depends on what we're trying to accomplish. In general, L<sub>1</sub> regularization often produces sparse weights (i.e., weights with many 0's), and L<sub>1</sub> regularization usually produces small distributed weights.

What this means is, when we are sure that our features are independent of each other and each has an effect on the label, it is often wise to use L<sub>2</sub> regularization, since no weights will be discounted. On the other hand, if our features are not independent of each other or if many weights don't have an effect on the prediction, then L<sub>1</sub> regularization is likely the better approach.
