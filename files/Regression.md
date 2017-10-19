Introduction to Regression
==========================

Regression is one of the most important topics of machine learning, for its simplicity and wide usage. For our purposes, the topic can be subdivided into two subtopics: 1) *linear regression* and 2) *logistic regression* (the latter of which could have been more aptly named binary regression). More general regression (*e.g.* polynomial) is also used in practiced, although not covered in this elementary introduction.

Training Data and Test Data
---------------------------
When creating ML models, there are two sets of data: 1) *labeled data*, or data for which we already have a correct classification, and 2) newly arriving data, or *unlabeled data*, for which it is our responsibility to predict a classification. The labeled, pre-classified data from which we generate our model is called the *training dataset*. When a subset of the labeled data is separated and used for model performance testing instead of for training the model, the term *test dataset* is used.

What is Regression?
-------------------
Regression is a modeling technique by which a predictive function is derived to classify all future incoming data. Specifically, if we have a training dataset in ℝ<sup>n</sup>, we derive a modeling function *f*: ℝ<sup>n</sup> → ℝ that approximates the training dataset. New data lacking classification can be plugged into *f* for prediction.

The Loss Function
------------------
When we create a model from the training data, how closely does the *training data* fit the model? It is the responsibility of the *loss function*, or *cost function*, **L**(f), to inform us how much information is lost (on the training set) by using the predictive model *f*. As the reader can probably guess, the training models with machine learning often involves a goal of minimizing the loss function.

A loss function often used in ML is the *mean square error (MSE)*. Minimizing the MSE is a very common technique to perform linear regression. If *x<sub>i</sub>* represents the training dataset of *n* points with respective classifications *y<sub>i</sub>*, the mean square error for the model *f* is defined to be:<p align="center">
  <img src="https://imgur.com/2vbsrTc.png" height="70">
</p>

*Questions:*
1. A training set is composed of the following data points: (*1,2*), (*2,3*), and (*3,6*). What is MSE(*f*) for *f(x) = 2x*?

Model Performance and Overfitting
---------------------------------
A natural question arises when the subject comes up of training data being used to create a model. How do we measure how well the model performs on incoming data? Since new data lacks classification, we cannot tell if the model is correctly classifying it or not.

A naive first attempt might be to find a continuous function that minimizes the loss function. There are a few problems with this approach:
1. There are many continuous functions that produce zero loss, and it would be impossible to choose between them. For example, if our training set is (*1*,*2*,*3*) with respective labels (*1*,*2*,*3*), we might choose *f(x) = x* as our predictive function since it produces zero loss. But a function that ascends from 1 to 10 and back down to 2 between 1 and 2 is also continuous with zero loss.
2. To fit the data exactly or near-exactly with a continuous function is often quite easy. However, fitting functions are often undulant and unnatural. A model should be simple, natural, and (for our purposes) not computation-intensive. This is because complex, twisting models rarely fit incoming data well.

*Example*

We have a training dataset composed of 3 points: *(-0.125,-0.25)*, *(-0.25,-0.5)*, and *(-0.7, -0.75)*.<p align="center">
  <img src="https://imgur.com/hrGriGK.png" height="130">
</p>

We can use two separate approaches to create a model by minimizing the MSE: *a*) Minimize the MSE over all polynomial functions, *b*) Minimize the MSE over all linear functions.

By approach *a*), we can minimize the MSE with the function *y = 3x<sup>6</sup>+2x<sup>4</sup>-x<sup>3</sup>-x<sup>2</sup>+2x*. In fact, the MSE becomes 0 with this function.<p align="center">
  <img src="https://imgur.com/uZ7BB6b.png" height="135">
</p>

By approach *b*, we can minimize the MSE with the function *y = 0.8x - 0.2*. While the MSE is low here, it is not 0.<p align="center">
  <img src="https://imgur.com/9CWYGXV.png" height="130">
</p>

Which model is better? To answer this question, we should ask ourselves how well each model *generalizes* to test data. If *x=-1* is an incoming data point, which makes more sense as a predicted value: *y=3* or *y=-1*? The first corresponds to the polynomial solution, while the latter corresponds to the linear solution. The difference is even more pronounced as we diverge from the range between -1 and 0.

The example above describes the concept of *overfitting*. A more complex function minimizes the loss function for the training data, but at the cost of being a poor predictor for new data. The way to avoid overfitting is by constraining solutions to a *hypothesis space* of functions. In the case above, it may have been wise to choose the space of linear functions as our hypothesis space. The study of techniques for narrowing the hypothesis space and avoiding overfitting is called *regularization*, to be covered later.

The question then naturally arises: from a testing perspective, how can we test if our model performs as expected and does not overfit the training data?

The answer is surprisingly simple, and has already been hinted at above with the definition of the test dataset. We only have labels for the training data, but we can subdivide the training data into 10 equal parts, then use 1 of these parts for training the model and 1 of these parts for testing the accuracy of the model. We can then switch out the training and test partitions, re-running the accuracy tests and averaging results for as many times as seems suitable. This approach is particularly desirable when a dataset is so large that it's intractable to process all at once for accuracy testing.

For example, if we have 1000 data points, we can shuffle this data into 10 equal partitions of 100 data points each. For the first test, we can choose partition 1 for training and partition 2 for testing. For the second test, we can choose partition 3 for training and partition 4 for testing. We can then average these accuracy measurements across tests to ensure that the process for generating the model is suitably precise.

Linear Regression
-----------------
*Linear regression* in ℝ is a type of regression that attempts to model data along the line *y = ax + b* for suitable *a* and *b* in ℝ. In our case of *m* data points in ℝ<sup>n</sup>, *a* is no longer in ℝ but is an (*m* x *n*)-dimensional matrix **A**, and *y* is an (*m* x 1) column matrix.

To simplify things, let us ignore the intercept *b* for a moment, and use **w** in place of *x* (to be clarified soon). In this case, we have the equation **A** **w** = **y**.

Before attempting to solve for **w** (noting above that **A** and **y** are provided by the training dataset), let's take a step back and ask what **w** actually is. **w** is an (*n* x 1)-dimensional column matrix. Remember that we are working in ℝ<sup>n</sup>, implying that our dataset is *n*-dimensional. Since the data point matrix **A** is multiplied by **w** to get the label **y**, **w** must be the *weight* applied to each dimension of the data point in order to apply the correct label, hence the renaming. Once we have determined **w**, we can predict future labels via the equation:<p align="center">
  <img src="https://imgur.com/sQdoVFk.png" height="40">
</p>

This is a solvable equation when *n* = *m* as long as **A** and **A<sup>T</sup>** are both invertible: In fact, **w** = (**A** **A<sup>T</sup>**)<sup>**-1**</sup> **A<sup>T</sup>**. The proof is as follows (with y-*hat* representing predicted values):<p align="center">
  <img src="https://imgur.com/Jm1bAdc.png" height="505">
</p>

Now, consider what happens when we also add an intercept *b*. Since this is just the addition of a constant, we can just expand each data point **x** by adding a 1 at the very front: **x** = [1, x<sub>1</sub>, ..., x<sub>m</sub>]<sup>T</sup>. This means that **w** will also increase at the front with a w<sub>0</sub> term representing the intercept: **w** = [w<sub>0</sub>, w<sub>1</sub>, ..., w<sub>m</sub>].

Logistic Regression
-------------------
Linear regression is a form of *continuous* regression: we map test data into a continuous range. We can also do a form of *discrete* regression, which is performed via logistic regression.

Logistic regression is a binary classification method -- a point can be classified into one of two categories with respective probabilities *p* and *1-p*.

Logistic regression is done via the *logistic function* (also called the *logistic sigmoid*), which is defined by:<p align="center">
  <img src="https://imgur.com/YHGlIgq.png" height="55">
</p>
, where <p align="center">
  <img src="https://imgur.com/f4VwdEX.png" height="32">
</p>
and <p align="center">
  <img src="https://imgur.com/GU358aS.png" height="32">

For the one-dimensional case, the logistic function is graphed as follows. The black function denotes β<sub>0</sub> = 0, β<sub>1</sub> = 0, blue represents the case β<sub>0</sub> = 1, β<sub>1</sub> = 0, and red represents the case β<sub>0</sub> = 0, β<sub>1</sub> = 5.

<p align="center">
  <img src="https://imgur.com/yST3e9i.png" height="90">
</p>

The reason this represents a form of *discrete regression* is that the range can be split into 2 equal parts  at *y = 0.5*. Note that the function is vertically symmetric around this point. Test data mapping to a value less than 0.5 can be grouped into category 1 while data mapping to a value >0.5 can be grouped into category 0.

Such a categorization is useful because we often want to classify into binary categories. For example, an ad system might want to predict whether or not a user with certain demographics will click on a specific ad.

As in the case of linear regression, our goal is to find a series of weights (written above as **w**<sup>**T**</sup>) that minimize the loss function. This is a complex process that will be clarified later.

Logistic regression can be generalized to more than 2 categories. This is called *multinomial logistic regression*.

The Iris Dataset
----------------

An historically famous dataset constructed in the 30s, the *Iris Dataset* is a small set of data points (150 entries) of species of the Iris plant, together with measurements of individual plants' sepal and petal length/widths. We can express the entries of the dataset using the following Scala case class:

```
// An enumeration-like Scala-specific construct.
trait Species
case object Setosa      extends Species
case object Versicolor  extends Species
case object Virginica   extends Species

case class Iris(
  sepalLength:  Float,
  sepalWidth:   Float,
  petalLength:  Float,
  petalWidth:   Float,
  species:      Species)
  ```

The Iris dataset is available [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

*Task:*

Create a model with Spark MLlib so that species can be predicted given sepal length/width and petal length/width. How well does your model perform?
