from nn_layers import input_layer, hidden_layer, output_layer
import numpy as np

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
    
    
class nn:
    
    def __init__(self, inputData, outData, reg_lambd ,output_nodes=None, output_f=softmax, loss_f=cross_entropy_loss, regularization_f=l2_regular):
        self.loss = -1 # Not Possible
        self.epoch = 0
        self.outData = outData
        
        self.reg_lambd = reg_lambd
        self.loss_f = loss_f
        self.regularization_f = regularization_f
        
        if (output_nodes is None):
            output_nodes = len(frozenset(outData))
        
        inputl = input_layer(inputData)
        outputl = output_layer(inputl,output_nodes, output_f)
        self.layers_list = [inputl,outputl]
        
    def append_hidden_layer(self, layer_nodes, layer_function):
        prevl = self.layers_list[-2]
        hiddenl = hidden_layer(prevl, layer_nodes, layer_function)
        self.layers_list[-1].set_prev(hiddenl)
        self.layers_list.insert(-1,hiddenl)
        
    def forward_prop(self):
        for i in self.layers_list[1:]:
            i.forward_prop()
            
    def calc_end_losses(self):
        data_loss = self.loss_f(self.get_output(),self.outData)
        reg_loss = sum([self.regularization_f(i.get_weights(),self.reg_lambd) for i in self.layers_list[1:]])
        self.loss = data_loss + reg_loss
        if (self.epoch % 100):
            print("Loss for epoch {} is {}".format(self.epoch,self.loss))
        
    def get_loss(self):
        return self.loss
            
    def print_wieghts(self):
        for i in self.layers_list[1:]:
            print(i.get_wieghts())
            
    def get_output(self):
        return self.layers_list[-1].get_signal()
    
    def print_signals(self):
        np.set_printoptions(suppress=True)
        for i in self.layers_list:
            print(i.get_signal())