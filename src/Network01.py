# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 21:46:17 2017

@author: longt
"""

import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
class Network:
    def __init__(self,sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
    def feedforward(self,a):
        """Return the output of the nentwork if "a" is input"""
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data: 
            n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size]
            for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                 print("Epoch {0} complete".format(j))
    
    def update_mini_batch(self,mini_batch,eta):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights=[w-(eta/len(mini_batch))*nw for w,nw in 
        zip(self.weights,nabla_w)]
        self.biases=[b-(eta/len(mini_batch))*nb for b,nb in
        zip(self.biases,nabla_b)]
    
    def backprop(self,x,y):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        a=x
        acti_layer=[x]
        zc=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,a)+b
            zc.append(z)
            a=sigmoid(z)
            acti_layer.append(a)
        

        #backward pass
        delta=self.cost_derivative(acti_layer[-1],y)*\
        self.sigmoid_derivative(zc[-1])
        
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,acti_layer[-2].T)
        #backward pass for cross-entropy
        '''
        delta=self.cost_derivative(acti_layer[-1],y)
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,acti_layer[-2].T)
         '''
        for l in range(2,self.num_layers):
            delta=np.dot(self.weights[-l+1].T,delta)*\
            self.sigmoid_derivative(zc[-l])
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,acti_layer[-l-1].T)
        return (nabla_b,nabla_w)
    
    # for squaristic cost function
    def cost_derivative(self,out_activations,y):
        return (out_activations-y)
    
    def sigmoid_derivative(self,z):
        return sigmoid(z)*(1-sigmoid(z))
        
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        
import mnist_loader      
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net=Network([784, 30, 10])
net.SGD(list(training_data), 30, 1, 0.5, list(test_data))
        