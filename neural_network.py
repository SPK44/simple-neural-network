from nn_layers import input_layer, hidden_layer, output_layer
import numpy as np
import math

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy_loss(x,y):
    num_examples = x.shape[1]
    correct_logprobs = -np.log(x[y, range(num_examples)])
    return np.sum(correct_logprobs)/num_examples

def l2_regular(w,lambd):
    return 0.5*lambd*np.sum(w*w)

def d_cross_entropy_loss(prev_delta, signal, y):
    deltas = np.dot(prev_delta, signal)
    deltas[y, range(signal.shape[1])] -= 1
    return deltas / signal.shape[1]

def d_tanh(output):
    return 1 - np.tanh(output)**2

def d_l2_regular(w,lambd):
    return lambd*w
    
class nn:
    
    def __init__(self, inputData, outData, output_nodes=None, output_f=softmax, loss_f=cross_entropy_loss, d_loss_f=d_cross_entropy_loss, regularization_f=l2_regular, d_regularization_f=d_l2_regular):
        self.loss = -1 # Not Possible
        self.epoch = 0
        self.full_outData = outData
        self.full_inputData = inputData
        self.outData = outData
        
        # default values, use train to modify
        self.learning_rate = 1e-3
        self.reg_lambd = 1e-3
        
        self.loss_f = loss_f
        self.d_loss_f = d_loss_f
        self.regularization_f = regularization_f
        self.d_regularization_f = d_regularization_f
        
        if (output_nodes is None):
            output_nodes = len(frozenset(outData))
        
        inputl = input_layer(inputData)
        outputl = output_layer(inputl,output_nodes, output_f)
        self.layers_list = [inputl,outputl]
        
    def append_hidden_layer(self, layer_nodes, layer_function=np.tanh, d_layer_function=d_tanh):
        prevl = self.layers_list[-2]
        hiddenl = hidden_layer(prevl, layer_nodes, layer_function, d_layer_function)
        self.layers_list[-1].set_prev(hiddenl)
        self.layers_list.insert(-1,hiddenl)
        
    def set_mini_batch(self, batch):
        examples = self.full_outData.shape[0]
        
        if(batch == 0):
            batch = examples #whole batch
            
        all_ex = np.arange(examples)
        np.random.shuffle(all_ex)
        ex_to_use = all_ex[:batch]
        batchx = self.full_inputData[ex_to_use]
        batchy = self.full_outData[ex_to_use]

        self.layers_list[0].set_input(batchx)
        self.outData = batchy
        
    def train(self, epochs, learning_rate, reg_lambd, batch=0):
        
        self.learning_rate = learning_rate
        self.reg_lambd = reg_lambd
        
        for epoch in range(epochs):
            self.set_mini_batch(batch)
            self.forward_prop()
            self.backward_prop()
            if (self.epoch % (epochs/20) == 0):
                print("Loss for epoch {} is {}".format(self.epoch, self.get_loss()))
            self.epoch += 1
            
    def test(self, testData):
        self.layers_list[0].set_input(testData)
        self.forward_prop()
        
    def k_fold_cross_validation(self,k=7, batch=1):
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
        folds = []
        
        for i in range(k):
            folds.append(all_ex[start:end])
            start += fold_size_list[k]
            end += fold_size_list[k]
            
        for i in range(k):
            ex_to_use = all_ex[folds[i]][:batch]
            batchx = self.full_inputData[ex_to_use]
            batchy = self.full_outData[ex_to_use]       
        
    def forward_prop(self):
        for i in self.layers_list[1:]:
            i.forward_prop()
           
    def backward_prop(self):
        prev_delta = self.d_loss_f(1, self.get_output(), self.outData)
        prev_weights = self.layers_list[-1].get_weights()
        self.layers_list[-1].set_delta(prev_delta)
        
        for i in reversed(self.layers_list[1:-1]):
            prev_delta = i.calc_delta(prev_delta,prev_weights)
            prev_weights = i.get_weights()
            
        for i in self.layers_list[1:]:
            i.update_weights(self.learning_rate, self.d_regularization_f(self.reg_lambd, i.get_weights()))        
        
    def get_loss(self):
        data_loss = self.loss_f(self.get_output(),self.outData)
        reg_loss = sum([self.regularization_f(i.get_weights(),self.reg_lambd) for i in self.layers_list[1:]])
        self.loss = data_loss + reg_loss
        return self.loss
            
    def get_weights(self):
        weight_list = []
        for i in self.layers_list[1:]:
            weight_list.append(i.get_weights())
        return weight_list
    
    def get_signals(self):
        signal_list = []
        for i in self.layers_list:
            signal_list.append(i.get_signal())
        return signal_list
        
    def get_output(self):
        return self.layers_list[-1].get_signal()