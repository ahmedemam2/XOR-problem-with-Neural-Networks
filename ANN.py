import Matrix
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1-y)


class NN:


    def __init__(self,input_nodes,hidden_nodes,output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_ih = np.random.uniform(-1, 1, (self.hidden_nodes,self.input_nodes))
        self.weights_ho = np.random.uniform(-1, 1, (self.output_nodes,self.hidden_nodes))
        self.biash = np.random.uniform(-1, 1, (self.hidden_nodes))
        self.biaso = np.random.uniform(-1, 1, (self.output_nodes))
        self.lr = 0.1

    def Predict(self,inputs):
        htemp = np.dot((self.weights_ih), inputs)

        hiddens = np.add(htemp, self.biash)

        hiddens = sigmoid(hiddens)

        outtemp = np.dot((self.weights_ho), hiddens)

        outputs = np.add(self.biaso, outtemp)

        guess = sigmoid(outputs)
        return guess

    def train(self,inputs,targets):


        htemp = np.dot((self.weights_ih),inputs)


        hiddens = np.add(htemp,self.biash)

        hiddens = sigmoid(hiddens)

        outtemp = np.dot((self.weights_ho),hiddens)

        outputs = np.add(self.biaso,outtemp)

        guess = sigmoid(outputs)





        # Calculate error
        # # Error = desired output - output
        outputs = guess
        outputerror = np.subtract(targets, outputs)


        # calculate gradient
        outputgradient = dsigmoid(outputs)

        outputgradient = np.dot((outputgradient),outputerror)
        outputgradient = np.multiply(self.lr,outputgradient)
        # calculate hiddenouput detlas
        hiddensT = np.transpose(hiddens)
        weightsHOdeltas = np.dot((outputgradient),hiddensT)
        # adjust weights by deltas
        self.weights_ho = np.add(weightsHOdeltas,self.weights_ho)
        # adjust bias by deltas
        self.biaso = np.add(self.biaso,outputgradient)
        # transpose since you are backpropagating
        # # Calculate Hidden layer errors
        weightsHO = np.transpose(self.weights_ho)
        hiddenerrors = np.dot(weightsHO,outputerror)
        # calculate hidden gradient
        hiddengradient = dsigmoid(hiddens)
        hiddengradient = np.dot((hiddengradient),hiddenerrors)
        hiddengradient = np.multiply(self.lr,hiddengradient)

        # caclulate inputhidden deltas
        inputsT = np.transpose(inputs)
        weightsIHdeltas = np.dot(hiddengradient,inputsT)
        self.weights_ih = np.add(weightsIHdeltas, self.weights_ih)
        self.biash = np.add(self.biash,hiddengradient)