from nn_layers import input_layer, hidden_layer, output_layer
import numpy as np
import math

def softmax(x):
    """
    Compute softmax values for each sets of scores in x
    
    Args:
        x: Numpy array to compute sofmax of
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy_loss(x,y):
    """
    Computes cross entropy loss
    
    Args:
        x: Numpy array of predictions
        y: Correct labels(enumerated version)
    """
    num_examples = x.shape[1]
    correct_logprobs = np.log(x[y, range(num_examples)])
    return -np.sum(correct_logprobs)/num_examples

def l2_regular(w,lambd):
    """
    l2 Regularization
    
    Args:
        w: Wieght Matrix
        y: Regularation Rate
    """
    return 0.5*lambd*np.sum(w*w)

def d_cross_entropy_loss(prev_delta, signal, y):
    """
    Derivative of Cross Entropy Loss
    
    Args:
        prev_delta: Gradient from previous layer(Should be 1 if output)
        signal: The output of the cross entropy loss
        y: The correct labels(enumerated)
    """
    deltas = np.dot(prev_delta, signal)
    deltas[y, range(signal.shape[1])] -= 1
    return deltas / signal.shape[1]

def d_tanh(output):
    """
    Derivative of Hyberbolic Tangent
    
    Args:
        output: The output of tanh on that layer
    """
    return 1 - np.tanh(output)**2

def d_l2_regular(w,lambd):
    """
    Derivative of l2 Regularization
    
    Args:
        w: Wieght Matrix
        y: Regularation Rate
    """
    return lambd*w
    
class nn:
    
    def __init__(self, inputData, outData, output_nodes=None, output_f=softmax, loss_f=cross_entropy_loss, d_loss_f=d_cross_entropy_loss, regularization_f=l2_regular, d_regularization_f=d_l2_regular):
        """
        Initializing the neural network
        
        Args:
            inputData: Training examples
            outData: Labels for training examples
            output_nodes: Specificy how many classess (Default, the program will guess based on labels)
            output_f: This is the activation function used in the output layer (Default: Softmax)
            loss_f: A callable function to compute the loss(Default: cross entropy loss)
            d_loss_f: Derivative of loss function
            regularization_f: Regularization Function
            d_regularization_f: Derivative of Regularization Function
        """
        
        self.loss = -1 # Init
        self.epoch = 0
        self.full_outData = outData
        self.full_inputData = inputData
        self.outData = outData
        
        # default values, use train or cross_val to modify
        self.learning_rate = 1e-3
        self.reg_lambd = 1e-5
        
        self.loss_f = loss_f
        self.d_loss_f = d_loss_f
        self.regularization_f = regularization_f
        self.d_regularization_f = d_regularization_f
        
        if (output_nodes is None):
            output_nodes = len(frozenset(outData))
        
        inputl = input_layer(inputData)
        outputl = output_layer(inputl,output_nodes, output_f)
        self.layers_list = [inputl,outputl] # Only two layers to start
        
    def append_hidden_layer(self, layer_nodes, layer_function=np.tanh, d_layer_function=d_tanh):
        """
        Adds a new layer to the network. This will always add the layer right before the output
        
        Args:
            layer_nodes: How many neaurons on the layer
            layer_function: Actication Function
            d_layer_function: Derivative of Activation Function
        """
        prevl = self.layers_list[-2]
        hiddenl = hidden_layer(prevl, layer_nodes, layer_function, d_layer_function)
        self.layers_list[-1].set_prev(hiddenl)
        self.layers_list.insert(-1,hiddenl)
        
    def set_mini_batch(self,inD,outD, batch):
        """
        Sets a certain amount of examples from certain data to be the mini-batch or SGD
        
        Args:
            inD: Input data to pick from
            outD: Output data that corresponds to input
            batch: Size of Batch
        """
        examples = outD.shape[0]
        
        if(batch == 0):
            batch = examples #whole batch
            
        # Select Examples    
        ex_to_use = np.random.choice(examples, batch, replace=False)
        batchx = inD[ex_to_use]
        batchy = outD[ex_to_use]

        self.layers_list[0].set_input(batchx)
        self.outData = batchy
        
    def train(self, epochs, learning_rate, reg_lambd, batch=0):
        """
        Method used to train the network
        
        Args:
            epochs: Number of epochs
            learning_rate: Network wide learning rate
            reg_lambd: Strength of regularization
            batch: Size of mini-batch, default to whole input
        """
        self.learning_rate = learning_rate
        self.reg_lambd = reg_lambd
        
        for epoch in range(epochs):
            self.set_mini_batch(self.full_inputData,self.full_outData, batch)
            self.forward_prop()
            self.backward_prop()
            if (self.epoch % (epochs/20) == 0):
                print("Loss for epoch {} is {}".format(self.epoch, self.get_full_loss()))
            self.epoch += 1
            
    def test(self, testData):
        """
        Method used get predictions for a certain set of examples
        
        Args:
            testData: Set of examples
        """
        self.layers_list[0].set_input(testData)
        self.forward_prop()
        
    def k_fold_cross_validation(self, epochs, learning_rate, reg_lambd, k=7, batch=1):
        """
        Method used for cross validation of a certain set of training examples. This is just
        regular random selection from k-folds
        
        Args:
            epochs: Number of epochs
            learning_rate: Network wide learning rate
            reg_lambd: Strength of regularization
            k: Amount of folds of the data
            batch: Size of mini-batch, default to whole input
            
        Returns:
            Average loss over k-folds
        """
        
        # Generated a list of examples to use
        examples = self.full_outData.shape[0]
            
        fold_size = examples // k
        last_fold_size = examples - fold_size * (k-1)
        fold_size_list = []
        
        for i in range(k-1):
            fold_size_list.append(fold_size)
        fold_size_list.append(last_fold_size)
                
        all_ex = np.arange(examples)
        np.random.shuffle(all_ex)
        
        start = 0
        end = fold_size
        
        self.learning_rate = learning_rate
        self.reg_lambd = reg_lambd
        
        sumL = 0
        
        for i in range(k):
            self.clear_weights()
            
            test_data = all_ex[start:end]
            # One liner to get the converse of a numpy array
            not_test_data = np.in1d(np.arange(len(all_ex)),test_data,assume_unique=True,invert=True)
            batchx = self.full_inputData[not_test_data]
            batchy = self.full_outData[not_test_data]

            self.epoch = 0
            
            for j in range(epochs):
                self.set_mini_batch(batchx,batchy, batch)
                self.forward_prop()
                self.backward_prop()
                self.epoch += 1
            
            self.outData = batchy
            self.test(batchx)
            tLoss = self.get_loss()
                
            self.outData = self.full_outData[test_data]
            self.test(self.full_inputData[test_data])    
            sumL += self.get_loss()
            print("Test Loss for Fold {} is {} :: Training Loss for Fold {} is {}".format(i,self.get_loss(),i,tLoss))
            
            start += fold_size_list[i]
            end += fold_size_list[i]
            
        return sumL / k
            
        
    def forward_prop(self):
        '''
        Does forward propagation of all of the layers
        '''
        for i in self.layers_list[1:]:
            i.forward_prop()
           
    def backward_prop(self):
        '''
        Does backward propagation across all of the layers
        '''
        prev_delta = self.d_loss_f(1, self.get_output(), self.outData)
        prev_weights = self.layers_list[-1].get_weights()
        self.layers_list[-1].set_delta(prev_delta)
        
        for i in reversed(self.layers_list[1:-1]):
            prev_delta = i.calc_delta(prev_delta,prev_weights)
            prev_weights = i.get_weights()
            
        for i in self.layers_list[1:]:
            i.update_weights(self.learning_rate, self.d_regularization_f(self.reg_lambd, i.get_weights()))      
            
    def clear_weights(self):
        '''
        Reinitializes the wieghts on every layer. Useful for cross validation
        '''
        for i in self.layers_list[1:]:
            i.reinit_weights()
        
    def get_loss(self):
        '''
        Gets the loss of the network using the current training examples, and current labels. For example the batch
        
        Returns:
            (By default) Cross entropy loss of data
        '''
        data_loss = self.loss_f(self.get_output(),self.outData)
        reg_loss = sum([self.regularization_f(i.get_weights(),self.reg_lambd) for i in self.layers_list[1:]])
        self.loss = data_loss + reg_loss
        return self.loss
    
    def get_full_loss(self):
        '''
        Gets the loss of the network using the full set of training examples, and complete labels
        
        Returns:
            (By default) Cross entropy loss of all of the data
        '''
        self.outData = self.full_outData
        self.test(self.full_inputData)
        return self.get_loss()
            
    def get_weights(self):
        '''
        For debugging; returns weights from every layer
        
        Returns:
            Numpy array of all weights
        '''
        weight_list = []
        for i in self.layers_list[1:]:
            weight_list.append(i.get_weights())
        return weight_list
    
    def get_signals(self):
        '''
        For debugging; returns signals from every layer
        
        Returns:
            Numpy array of all signals
        '''
        signal_list = []
        for i in self.layers_list:
            signal_list.append(i.get_signal())
        return signal_list
        
    def get_output(self):
        '''
        COnvienence method to get the output of the last layer
        
        Returns:
        '''
        return self.layers_list[-1].get_signal()