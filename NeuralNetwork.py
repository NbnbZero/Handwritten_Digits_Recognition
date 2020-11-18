import numpy as np
import sys
import matplotlib.pyplot as plt

# number of input, hidden and output nodes
inputNodesNum = 784
hidden1NodesNum = 100
hidden2NodesNum = 30
outputNodesNum = 10

# learning rate is 0.1
learning_rate = 0.1

# batch size = 100
batch_size = 100

# number of epochs = 10
epochNum = 10

# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# sigmoid derivative
def sigmoid_derivative(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

class NeuralNetwork:

    def __init__(self, inputNodesNum, hidden1NodesNum, hidden2NodesNum, outputNodesNum):

        # weights initialization
        weightIH = np.random.normal(0.0, 1/np.sqrt(inputNodesNum), (hidden1NodesNum,inputNodesNum))
        weightHH = np.random.normal(0.0, 1/np.sqrt(hidden1NodesNum), (hidden2NodesNum,hidden1NodesNum))
        weightHO = np.random.normal(0.0, 1/np.sqrt(hidden2NodesNum), (outputNodesNum,hidden2NodesNum))
        self.weights = [weightIH, weightHH, weightHO]

    def forwardPassing(self, singleInput, singleLabel):
        x = singleInput
        rawInputs = []
        activated = []
        activated.append(x)
        # for hidden layers we use sigmoid activation function
        for weight in self.weights[:-1]:
            # z = weight * x
            z = np.dot(weight, x)
            rawInputs.append(z)
            # activating inputs
            x = sigmoid(z)
            activated.append(x)
        # for output layer we use softmax activation function
        z = np.dot(self.weights[-1], x)
        rawInputs.append(z)
        final_output = softmax(z)
        activated.append(final_output)

        partial_derivative = - (singleLabel - final_output)
        return rawInputs, activated, partial_derivative

    def backwardPropagation(self, deltaE, rawInputs, activated, partial_derivative):
        # for output layer, delta_E_HO is set to oi * (-(t-oj))
        outputPD = partial_derivative   # partial derivative for output layer
        deltaE[-1] = np.dot(outputPD, activated[-2].T)   # update delta E

        # for hidden layer 2, delta_E_HH is set to oi * f'(inj) * sum(wk * output_partial_derivative)
        hidden2Input = rawInputs[-2]
        h2PD = np.dot(self.weights[-1].T, outputPD) # partial derivative for hidden layer 2
        h2PD *= sigmoid_derivative(hidden2Input)
        deltaE[-2] = np.dot(h2PD, activated[-3].T)

        # for hidden layer 1, delta_E_IH is set to oi * f'(inj) * sum(wk * f'(ink) * hiddenlayer2_partial_derivative)
        hidden1Input = rawInputs[-3]
        h1PD = np.dot(self.weights[-2].T, h2PD) # partial derivative for hidden layer 1
        h1PD *= sigmoid_derivative(hidden1Input)
        deltaE[-3] = np.dot(h1PD, activated[-4].T)

        return deltaE

    def trainingProcess(self, singleInput, singleLabel):
        # create a list for delta E
        deltaE = []
        for layer_weights in self.weights:
            deltaE.append(np.zeros(layer_weights.shape))

        # forward pass
        rawInputs, activated, partial_derivative = self.forwardPassing(singleInput, singleLabel)

        # back-propagation
        deltaE = self.backwardPropagation(deltaE, rawInputs, activated, partial_derivative)

        return deltaE


# create an neural network
network = NeuralNetwork(inputNodesNum,hidden1NodesNum,hidden2NodesNum,outputNodesNum)

# create an all-zero's 10*1 array except for target index being 1
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# load data
def load_data(training_images, training_labels, testing_images):
    training_input = []
    training_res = []
    testing_input = []

    with open(training_images,'r') as trImg:
        lines = trImg.readlines()
        for line in lines:
            img = np.asfarray(line.split(','))  # converting string array to float array
            training_input.append(np.reshape(img,(784,1))/256)

    with open(training_labels,'r') as trLab:
        lines = trLab.readlines()
        for label in lines:
            training_res.append(vectorized_result(int(label)))

    training_data = list(zip(training_input,training_res))  # [(array(float), label)]

    with open(testing_images, 'r') as tstImg:
        lines = tstImg.readlines()
        for line in lines:
            img = np.asfarray(line.split(','))
            testing_input.append(np.reshape(img,(784,1))/256)

    return training_data, testing_input

# split training data to batches
def splitData(training_data, training_size):
    res=[]
    for i in range(0, training_size, batch_size):
        tmp = training_data[i:i+batch_size]
        res.append(tmp)
    return res


def train(training_data):
    # training size
    training_size = len(training_data)
    # list of training accuracy and epoches used for learning curve
    training_accuracy = []
    epoches = []
    # in each epoch out of 10
    for epoch in range(epochNum):
        epoches.append(epoch+1)
        # split data into 100 baches
        baches = splitData(training_data, training_size)
        for bach in baches:
            # update weights
            deltaE = []
            for layer_weights in network.weights:
                deltaE.append(np.zeros(layer_weights.shape))

            for singleInput, singleLabel in bach:
                dt_E = network.trainingProcess(singleInput, singleLabel)
                tmpE = []
                for e1, e2 in zip(deltaE, dt_E):
                    tmpE.append(e1 + e2)
                deltaE = tmpE

            newWeight = []
            for old_weight, dt_E in zip(network.weights, deltaE):
                newWeight.append(old_weight - learning_rate * dt_E)
            network.weights = newWeight

        accuracy = getTrainingAccuracy(training_data, training_size)
        print("Accuracy on training epoch {}: {}%".format(epoch+1, accuracy))
        training_accuracy.append(accuracy)

    print("Training process complete!")
    plot_learningCurve(training_accuracy, epoches)

def plot_learningCurve(training_accuracy, epoches):
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.xlim((0,11))
    plt.ylim((10,100))
    plt.title('Training accuracy vs. Epoches')
    plt.plot(epoches, training_accuracy, marker='s', linestyle='-')
    plt.show()

# predict results and save to .csv file
def predict_res(testing_input):
    res = []
    for image in testing_input:
        input = image
        for weight in network.weights[:-1]:
            input = sigmoid(np.dot(weight, input))

        f_output = softmax(np.dot(network.weights[-1], input))
        res.append(np.argmax(f_output))
    result = np.int32(res)
    np.savetxt('test_predictions.csv', result, delimiter=',', fmt='%d')

# calculate training accuracy among each epoch
def getTrainingAccuracy(training_data, training_size):
    correctNum = 0;
    for image, label in training_data:
        input = image
        for weight in network.weights[:-1]:
            input = sigmoid(np.dot(weight, input))

        f_output = softmax(np.dot(network.weights[-1],input))
        if (np.argmax(f_output) == np.argmax(label)):
            correctNum += 1

    return correctNum * 100 / training_size

# input data from terminal command
training_images = "train_image.csv"
training_labels = "train_label.csv"
testing_images = "test_image.csv"
training_data, testing_input = load_data(training_images, training_labels, testing_images)

# train the data set
train(training_data)

# using testing_input to predict test results
predict_res(testing_input)