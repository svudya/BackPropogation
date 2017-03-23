#build a feed forward neural network
import numpy as np

def __sigmoid(z):
	return 1/(1+exp(-z))

def z_variable(theta, X):
	thetaT = np.transpose(np.matrix(theta))
	return np.matmul(theta,X)

def layer_matrix():
	np.random.seed(1)

	syn_weights_1 = 2*np.random.random((3,5))-1
	print syn_weights_1
	syn_weights_2 = 2*np.random.random((5,5))-1
	print syn_weights_2
	syn_weights_3 = 2*np.random.random((5,4))-1
	print syn_weights_3

	a1 = __sigmoid(z_variable(syn_weights_1, X))
	a2 = __sigmoid(z_variable(syn_weights_2, a1))
	y = __sigmoid(z_variable(syn_weights_3, a2))

if __name__ == "__main__":
	layer_matrix()