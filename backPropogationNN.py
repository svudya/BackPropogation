# Implement Backpropogation algorithm
# cost function for neural networks
# m = no of training examples
# n = no of attributes in input
# sl = no of units in 'l' layer
# L = no of layers present in a neural network
# K = no of output nodes
# lambda = regularization parameter

# implement back propogation for a 4 layer [3,5,5,4] neural network for one input x = [x1,x2,x3]and one ouput y = [y1,y2,y3,y4]
# here m = 1, n = 3, s1 = 3, s2 = 5, s3 = 5, s4 = 4, L = 4, K = 4, lambda = ?
import numpy as np

def __sigmoid(z):
	return 1/(1+np.exp(-z))
def __sigmoid_derivative(z):
	return (np.exp(-z)/(1-np.exp(-z)))

def z_variable(theta, X):
	thetaT = np.transpose(np.matrix(theta))
	print thetaT
	return np.matmul(thetaT,X)

def layer_matrix():
	X = np.matrix([1,2,3]).getT()
	Y = np.matrix([1,0,0,1]).getT()
	m = 1
	n = 3
	s1 = 3
	s2 = 5
	s3 = 5
	s4 = 4
	L = 4
	K = 4
	_lambda = 100

	np.random.seed(1)

	syn_weights_1 = 2*np.random.random((3,5))-1
	print syn_weights_1
	syn_weights_2 = 2*np.random.random((5,5))-1
	print syn_weights_2
	syn_weights_3 = 2*np.random.random((5,4))-1
	print syn_weights_3

	print Y

	z1 = z_variable(syn_weights_1, X)
	a1 = __sigmoid(z1)
	z2 = z_variable(syn_weights_2, a1)
	a2 = __sigmoid(z2)
	z3 = z_variable(syn_weights_3, a2)
	ay = __sigmoid(z3)

	print a1
	print a2
	print ay

	print "del time"
	#del1 = __sigmoid_derivative(z3)
	del1 = np.subtract(Y, ay)
	print del1
	del2 = np.multiply(np.matmul(np.matrix(syn_weights_3),del1), (np.multiply(a2,(1-a2))))
	print del2
	del3 = np.multiply(np.matmul(np.matrix(syn_weights_2),del2), (np.multiply(a1,(1-a1))))
	print del3
	print "Dtime"
	D1 = np.matmul(del1, ay.getT())
	print D1
	D2 = np.matmul(del2, a2.getT())
	print D2
	D3 = np.matmul(del3,a1.getT())
	print D3


def calculateCost(theta, X, Y):
	cost_per_unit = np.add(np.matmul(Y,np.log(__sigmoid(z_variable(theta,X)))),np.matmul(np.subtract(1,Y),np.log((__sigmoid(z_variable(theta,X))))))


if __name__ == '__main__':
	layer_matrix()


