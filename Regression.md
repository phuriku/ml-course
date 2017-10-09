Introduction to Regression
==========================

Regression is one of the most important topics of machine learning. For our purposes, the topic can be subdivided into two subtopics: 1) *linear regression* and 2) *logistic* regression (the latter of which could have been more aptly named binary regression). More general regression (e.g. quadratic) is also possible, although not covered in this elementary introduction.

What is Regression?
-------------------
Regression is a modeling method by which a predictive function is derived to be used to classify all future data points. Specifically, if we have a dataset of *n*-dimensional data points for each of which we know the classification, we can derive a generalized function *f*: ℝ<sup>n</sup> → ℝ that approximates the results of the training set data. Newly arriving data lacking classification can be plugged into *f* for prediction.

Training Data and Test Data
---------------------------
When creating ML models, there are two sets of data: 1) *labeled data*, or data for which we have a correct classification, and 2) newly arriving data, or unlabeled data for which it is our responsibility to predict a classification. The labeled, pre-classified data from which we generate our model is called the *training dataset*, while the newly arriving data for which we will use our generated model to predict a classification is called the *test dataset*.

The Loss Function
------------------
When we create a model from the training data, how closely does the *training data* fit the model? It is the responsibility of the *loss function*, or *cost function*, **L**(f), to inform us how much information is lost by using the predictive model *f*.

A common loss function is the *mean square error (MSE)*, and minimizing this cost function is a very commonly used way to do linear regression. If *x<sub>i</sub>* represents our dataset of *n* points with respective classifications *y<sub>i</sub>*, the mean square error for the model *f* is defined to be (1/*n*) Σ<sub>*n*</sub> (f(*x<sub>i</sub>*) - *y<sub>i</sub>*))<sup>2</sup>.

Question: If we have a training set of points (*1*,*2*,*3*) with respective labels of (*2*,*3*,*6*), what is MSE(*f*) for *f(x) = 2x*?

Overfitting and Model Performance
---------------------------------
A natural question arises when the subject comes up of training data being used to create a model. How do we measure how well the model performs on incoming (i.e. test) data? Since test data lacks classification, we cannot tell if the model is correctly classifying the test data or not.

A naive first attempt might be to find a continuous function that minimizes the loss function. There are a few problems with this approach:
1. There are many continuous functions that produce zero loss, and it would be impossible to choose between them. For example, if our training set is (*1*,*2*,*3*) with respective labels (*1*,*2*,*3*), we might choose *f(x) = x* as our predictive function since it produces zero loss. But a function that ascends from 1 to 10 and back down to 2 between 1 and 2 is also continuous with zero loss.
2. To fit the data exactly or near-exactly with a continuous function is often quite easy. However, the fitting function is often extremely undulant and unnatural. A model should be simple, natural, and (for our purposes) not computation-intensive -- complex, twisting models rarely fit test data well.

The second point above describes *overfitting*. It minimizes the loss function for the training data, but at the cost of not being a good predictor for test data.

Linear Regression
-----------------



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
