import numpy as np
class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize)
    def forward(self, X):
        self.z = np.dot(X, self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.w2)
        o = self.sigmoid(self.z3)
        return o
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
    def sigmoidPrime(self, s):
        return s*(1-s)
    def backward(self, X, Y, o):
        self.o_error = Y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.w1 += X.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)
    def train(self, X, Y):
        o = self.forward(X)
        self.backward(X, Y, o)
X = np.array(([2,9],[1,5],[3,6]), dtype=float) # [hours sleeping, hours studying]
Y = np.array(([92], [86], [89])) # [marks scored in test]
X = X/np.amax(X, axis=0)
Y = Y/100
NN = Neural_Network()
for i in range(700):                                         # Training the NN 1000 times
    print("Input:\n", str(X))
    print("Actual output:\n", str(Y))
    print("Predicted output:\n", str(NN.forward(X)))
    mean_sum_squared_loss = np.mean(np.square(Y - NN.forward(X)))
    print("Loss:\n", str(mean_sum_squared_loss))
    print("\n")
NN.train(X,Y)