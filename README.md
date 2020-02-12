# Artificial_Neural_Network_2_Layer

An Artificial neural network is created from the scratch. 
A two neural netowrk is implimented using Linear Algebra. The first layer consists of the following equation:

The solution is a linear algebra version of the Forward Propogation and Back Propogation concepts of deep learning.

* Forward Propogation- The first layer of the neural network encompasses of neurons represented by the W numpy array. x is the input array. The bias term is represented by the b1 array.

First Layer Linear Algebra		
		
	h1 = sigmoid(X.dot(W1) + b1) 

where the sigmoid function is defined as:

	1 / (1 + np.exp(-x))

Second Layer Linear Algebra:

	Z2 = A.dot(W2) + b2
	Y = softmax(Z2)

where the softmax function is defined as:

	expA = np.exp(A)
	expA / expA.sum(axis=1, keepdims=True)


* The back propogation equations are implemented as:

        delta2 = y_hat - Y_encoded
        delta1 = (delta2).dot(W2.transpose()) * h1 * (1 - h1)

        W2 -= alpha * h1.T.dot(delta2)
        b2 -= alpha * (delta2).sum(axis=0)

        W1 -= alpha * X.T.dot(delta1)
        b1 -= alpha * (delta1).sum(axis=0)

Alpha (the learning rate) has been set to a low value of 0.03 


Part 2:
The same ANN is used to classify the digits of MNIST dataset 
The raw data first split into 10 folds and pre-processing is done.
