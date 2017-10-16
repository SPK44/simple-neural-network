import numpy as np



class input_layer: 
        
    def __init__(self, inX):
        """ inX is the input data with any amount of examples, essentially the signal is returned, plus a bias"""
        examples = inX.shape[0]
        bias = np.full((examples, 1),1)
        self.x = (np.concatenate((bias, inX), axis=1)).T
        self.s = self.x

    def get_signal(self):
        return self.s
    
    def get_shape(self):
        return self.s.shape

    
class hidden_layer:
    
    def __init__(self, prev_layer, nodes, activation_f):
        self.nodes = nodes
        self.prev_layer = prev_layer
        self.activation_f = activation_f
        self.w = 0.01 * np.random.randn(prev_layer.get_shape()[0], nodes)
        
    def forward_prop(self):
        """ This will push the signals forward"""
        self.x = np.dot((self.w).T,self.prev_layer.get_signal())
        
        examples = self.prev_layer.get_shape()[1]
        bias = np.full((1, examples),1)
                
        self.s = np.concatenate((bias, self.activation_f(self.x)), axis=0)
        
    def set_prev(self, prev_layer):
        self.prev_layer = prev_layer
        
    def get_signal(self):
        return self.s
    
    def get_shape(self):
        return self.s.shape
    
    def get_weights():
        return self.w
    
class output_layer:
    
    def __init__(self, prev_layer, nodes, activation_f):
        self.nodes = nodes
        self.prev_layer = prev_layer
        self.activation_f = activation_f
        self.w = 0.01 * np.random.randn(prev_layer.get_shape()[0], nodes)
        self.s = np.empty((nodes,prev_layer.get_shape()[1]))
        
    def forward_prop(self):
        """ This will push the signals forward"""
        self.x = np.dot((self.w).T,self.prev_layer.get_signal())
        self.s = self.activation_f(self.x)
        
    def set_prev(self, prev_layer):
        self.prev_layer = prev_layer
        
    def get_signal(self):
        return self.s
    
    def get_shape(self):
        return self.s.shape
    
    def get_weights():
        return self.w