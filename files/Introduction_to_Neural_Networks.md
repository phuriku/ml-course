Introduction to Neural Networks
===============================

This page is meant as a brief introduction to neural networks and how to model them with Spark ML. The conceptual pedagogy is largely a summary of [this page]( http://neuralnetworksanddeeplearning.com/chap1.html), which is a gentle yet comprehensive introduction to neural networks. The amount of free literature on neural networks and artificial intelligence is surprisingly plentiful, which is why I've chosen not to invent the wheel here.

What is a neural network?
-------------------------
A neural network is a machine learning paradigm that is influenced by how neurons act in the human brain. This isn't to say that the modern neural networks work the same as the human brain -- actually, they are structured quite differently, and the implementations that are closer in reality to the way that human brains work (e.g. recurrent neural networks) haven't been as effective as more artificial implementations.

The human brain uses 86 billion neurons. Most applications using neural networks use far fewer than 1 million artificial neurons, and even many of these are so computationally intensive that they prove ineffective in terms of time or money. As such, the artificial neural networks currently in use come nowhere near to meeting the performance of the human brain, and are not expected to meet this level until at least 2050 (and even this is a somewhat ambitious date). The largest neural networks in use today are about as powerful as the brain of an ant or a cockroach. In this tutorial, we will use less than a hundred neurons to do a simple handwriting recognition task.

![Imgur](https://upload.wikimedia.org/wikipedia/commons/f/fe/Brain_size_comparison_-_Brain_neurons_%28billions%29.png)

What's wrong with regression?
-----------------------------

There's nothing inherently wrong with regression, and many problems can be solved by simple linear regression. However, it must be mentioned that there are implicit assumptions built into any regression model. In particular, due to the risk of overfitting, non-linear regression usually only tends to be useful for distributions known to be order 2. And of course, linear regression cannot model non-linear data sets.

As such, complex use cases like computer vision and speech recognition must look elsewhere. Neural networks have largely helped to fill this gap in complexity, even though they are still in their infancy and in no way compare to human intelligence in performance.

Perceptrons
-----------
To understand neural networks, we must first understand their elementary components. We must therefore first study perceptrons. (To hint at what is to come, the most simple type of neural network is called a multilayer perceptron.)

A perceptron is meant to conceptually model a neuron in its most simplistic form. A perceptron takes input parameters x_1, x_2, ..., x_n and produces an output value of either 1 or 0 (i.e., embodying the concept of either firing or not firing).

The logic here is that there exist implicit weights w_1, w_2, ..., w_n for each input parameter, and if the dot product of the inputs and weights (v dot w = sum v_i w_i) exceeds a particular threshold value *t*, the perceptron outputs value 1. If it doesn't exceed *t*, then the output is 0.

There is a simplification that we can make here. Instead of using the threshold value *t*, we can introduce a bias term *b*, and then make the comparison against 0:

![Imgur](https://i.imgur.com/IeJmyQc.png)

Artificial Neurons
---------------
Unfortunately, there is a practical issue at hand with perceptrons: training them can be difficult. This is because small variations in the input parameters can produce a large change in the output: the perceptron can only output values of 0 or 1. As such, it becomes advantageous in a practical sense to introduce a function that will represent values of 0 and 1 conceptually but is more flexibly variant.

This function is the logistic sigmoid that we have seen before in regression tasks:

The logistic sigmoid can take values in the range of 0 to 1. A value greater than 0.5 conceptually represents a 1, while a value less than 0.5 represents a 0. We recall that the logistic sigmoid takes the form

![Imgur](https://camo.githubusercontent.com/aa3a25c4107999a7ffbe6bc87a5e972b01dec615/68747470733a2f2f696d6775722e636f6d2f5948476c4967712e706e67)

. This of course presumes that the weights are represented as a row vector *w*.


Feedforward networks
---------------
A feedforward network is a generalization of the sigmoid neuron to multiple layers, many of which are "hidden" (meaning nothing more than that they're intermediary layers).

We can visualize a neural network as the following:

![Imgur](http://neuralnetworksanddeeplearning.com/images/tikz11.png)

The lines in the above diagram are all distinct weights, while the circles represent neurons. The values of neurons in layer m therefore single-handedly determine the values of the neurons in layer m+1, after the weights have been applied. This is the distinguishing feature of feedforward networks: each layer is dependent only on the previous layer; more specifically, there are no feedback loops.

 Another type of neural network, called a *recurrent neural network*, does incorporate the concept of feedback loops. While the recurrent network paradigm is closer to the human brain in functionality, it has thus far not produced performance correspondent to its increase in complexity, and has thus proved less useful in practice than feedforward neural networks, which we shall exclusively examine herein.

 Backpropagation
 ---------------

It should be clear, then, that training a model with a neural network means finding an appropriate value for each weight. This is done through the process of *backpropagation*, which in simple terms means going back and changing the weights appropriately after observing a piece of training data. Hence, when a piece of training data passes through the neural network, the weights are adjusted through backpropagation, and then the neural network moves on to evaluating the next piece of training data.

To do this, we first need a cost function. In this case, we use:

The precise algorithm for backpropagation is complex, and involves complex mathematics. As its name implies, the way it works is by approximating the errors contributed by weights at the last levels, making proper adjustments, and then propagating these adjustments backward to weights at earlier levels.

The MNIST Handwriting Dataset
=============================

The "Hello World" of neural networks is a handwriting recognition task that happens to be quite difficult to solve for non-neural network machine learning techniques but easily handled by a feedforward network. The MNIST handwriting dataset is a set of 60,000 correctly labeled handwritten digits from 0-9. Using this training dataset, we can create a neural network that can recognize any handwritten digit from 0-9.

Schema
------

We will first need to create a schema for creating the neural network and analyzing its performance. We will create the neural network with Spark ML's multilayer perceptron implementation (recall that multilayer perceptron is just another name for feedforward neural network). To analyze its performance, we will split the dataset into 50,000 training samples and 10,000 test samples.

The dataset is composed of 28x28 pixel images of scanned handwritten digits. As such, our feature space has dimension 784 = 28 x 28. The data is available in libsvm format (here: ), which is the format preferred by Spark for processing.
