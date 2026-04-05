import pandas as pd
import numpy as np
import pickle
import random

# set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class Network(object):
    def __init__(self, sizes, optimizer="sgd"):
        # weights connect two layers, each neuron in layer L is connected to every neuron in layer L+1,
        # the weights for that layer of dimensions size(L+1) X size(L)
        # the bias in each layer L is connected to each neuron in L+1, the number of weights necessary for the bias
        # in layer L is therefore size(L+1).
        # The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        self.weights = [((2/sizes[i-1])**0.5)*np.random.randn(sizes[i], sizes[i-1]) for i in range(1, len(sizes))]
        self.biases = [np.zeros((x, 1)) for x in sizes[1:]]
        self.optimizer = optimizer
        if self.optimizer == "adam":
            # adam optimizer
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.t = 0

    def train(self, training_data, training_class, val_data, val_class, epochs, mini_batch_size, eta, lmbda=0.0, lr_decay_k=None):
        # training data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # training_class - numpy array of dimensions [c x m], where c is the number of classes
        # epochs - number of passes over the dataset
        # mini_batch_size - number of examples the network uses to compute the gradient estimation

        iteration_index = 0
        eta_current = eta

        n = training_data.shape[1]
        for j in range(epochs):
            # Exponential learning rate decay: eta_t = eta * exp(-k * t)
            if lr_decay_k is not None:
                eta_current = eta * np.exp(-lr_decay_k * j)
            else:
                eta_current = eta
            print(f"Epoch {j}, Learning Rate: {eta_current}")
            loss_avg = 0.0
            mini_batches = [
                (training_data[:,k:k + mini_batch_size], training_class[:,k:k+mini_batch_size])
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                output, Zs, As = self.forward_pass(mini_batch[0])
                gw, gb = self.backward_pass(output, mini_batch[1], Zs, As)
                self.update_network(gw, gb, eta_current, lmbda, n)
                iteration_index += 1
                loss = cross_entropy(mini_batch[1], output)
                loss_avg += loss

            print("Epoch {} complete".format(j))
            print("Loss:" + str(loss_avg / len(mini_batches)))
            if j % 10 == 0 and val_data is not None and val_class is not None:
                self.eval_network(val_data, val_class)

    def eval_network(self, validation_data,validation_class):
        # validation data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # validation_class - numpy array of dimensions [c x m], where c is the number of classes
        n = validation_data.shape[1]
        loss_avg = 0.0
        tp = 0.0
        for i in range(validation_data.shape[1]):
            example = np.expand_dims(validation_data[:,i],-1)
            example_class = np.expand_dims(validation_class[:,i],-1)
            example_class_num = np.argmax(validation_class[:,i], axis=0)
            output, Zs, activations = self.forward_pass(example)
            output_num = np.argmax(output, axis=0)[0]
            tp += int(example_class_num == output_num)

            loss = cross_entropy(example_class, output)
            loss_avg += loss
        print("Validation Loss:" + str(loss_avg / n))
        print("Classification accuracy: "+ str(tp/n))

    def update_network(self, gw, gb, eta, lmbda=0.0, n=1):
        # gw - weight gradients - list with elements of the same shape as elements in self.weights
        # gb - bias gradients - list with elements of the same shape as elements in self.biases
        # eta - learning rate
        # lmbda - L2 regularization parameter
        # n - number of training examples
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= eta * (gw[i] + (lmbda / n) * self.weights[i])
                self.biases[i] -= eta * gb[i]
        else:
            # adam optimizer 
            self.t += 1
            for i in range(len(self.weights)):
                # L2 regularization
                gw_reg = gw[i] + (lmbda / n) * self.weights[i]
                # update biased first moment 
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gw_reg
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gb[i]
                # update biased second raw moment 
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gw_reg ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gb[i] ** 2)
                # compute bias-corrected first moment 
                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                # compute bias-corrected second moment
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                # update parameters
                self.weights[i] -= eta*m_w_hat/(np.sqrt(v_w_hat)+self.epsilon)
                self.biases[i] -= eta*m_b_hat/(np.sqrt(v_b_hat)+self.epsilon)


    def forward_pass(self, input):
        
        activation = input
        activations = [input]  # list to store all activations
        Zs = []  # list to store all z vectors, layer by layer
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activation)+self.biases[i]
            Zs.append(z)
            if i == len(self.weights) - 1:
                # output layer uses softmax activation
                activation = softmax(z)
            else:
                # hidden layers use sigmoid activation
                activation = sigmoid(z)
            activations.append(activation)
        return activations[-1], Zs, activations

    def backward_pass(self, output, target, Zs, activations):

        gw=[np.zeros(w.shape) for w in self.weights]
        gb=[np.zeros(b.shape) for b in self.biases]
        m=output.shape[1]

        # output layer 
        delta = softmax_dLdZ(output, target) 
        gw[-1]=np.dot(delta, activations[-2].T)/m
        gb[-1]=np.sum(delta, axis=1, keepdims=True)/m

        # backpropagate through hidden layers
        for l in range(2, len(self.weights)+1):
            z=Zs[-l]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].T, delta)*sp
            gw[-l]=np.dot(delta,activations[-l - 1].T)/m
            gb[-l]=np.sum(delta, axis=1, keepdims=True)/m

        return gw, gb

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def softmax_dLdZ(output, target):
    # partial derivative of the cross entropy loss w.r.t Z at the last layer
    return output - target

def cross_entropy(y_true, y_pred, epsilon=1e-12):
    targets = y_true.transpose()
    predictions = y_pred.transpose()
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z): 
    return sigmoid(z) * (1 - sigmoid(z))

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def load_data_cifar(train_file, test_file):
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)
    train_data = np.array(train_dict['data']) / 255.0
    train_class = np.array(train_dict['labels'])
    train_class_one_hot = np.zeros((train_data.shape[0], 10))
    train_class_one_hot[np.arange(train_class.shape[0]), train_class] = 1.0
    test_data = np.array(test_dict['data']) / 255.0
    test_class = np.array(test_dict['labels'])
    test_class_one_hot = np.zeros((test_class.shape[0], 10))
    test_class_one_hot[np.arange(test_class.shape[0]), test_class] = 1.0
    return train_data.transpose(), train_class_one_hot.transpose(), test_data.transpose(), test_class_one_hot.transpose()

if __name__ == "__main__":
    train_file = "./data/train_data.pckl"
    test_file = "./data/test_data.pckl"
    train_data, train_class, test_data, test_class = load_data_cifar(train_file, test_file)
    val_pct = 0.1
    val_size = int(train_data.shape[1] * val_pct)
    val_data = train_data[..., :val_size]
    val_class = train_class[..., :val_size]
    train_data = train_data[..., val_size:]
    train_class = train_class[..., val_size:]
    # The Network takes as input a list of the numbers of neurons at each layer. The first layer has to match the
    # number of input attributes from the data, and the last layer has to match the number of output classes
    # The initial settings are not even close to the optimal network architecture, try increasing the number of layers
    # and neurons and see what happens.
    # Example usage: try optimizer="adam" and lmbda=0.01 for L2 regularization
    # net = Network([train_data.shape[0], 64, 10], optimizer="sgd")

    net = Network([train_data.shape[0], 128, 10], optimizer="adam")
    # net.train(train_data, train_class, val_data, val_class, 40, 64, 0.01, lmbda=0.00)
    net.train(train_data, train_class, val_data, val_class, 40, 32, 0.0001, lmbda=0.0, lr_decay_k=0.001)
    net.eval_network(test_data, test_class)