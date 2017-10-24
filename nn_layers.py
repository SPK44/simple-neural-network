import numpy as np



class input_layer: 
        
    def __init__(self, inX):
        """ inX is the input data with any amount of examples, essentially the signal is returned, plus a bias"""
        examples = inX.shape[0]
        bias = np.full((examples, 1),1)
        self.x = (np.concatenate((bias, inX), axis=1)).T
        self.s = self.x
        
    def set_input(self,inX):
        examples = inX.shape[0]
        bias = np.full((examples, 1),1)
        self.x = (np.concatenate((bias, inX), axis=1)).T
        self.s = self.x

    def get_signal(self):
        return self.s
    
    def get_shape(self):
        return self.s.shape

    
class hidden_layer:
    
    def __init__(self, prev_layer, nodes, activation_f, d_activation_f):
        self.nodes = nodes
        self.prev_layer = prev_layer
        self.activation_f = activation_f
        self.d_activation_f = d_activation_f
        
        self.w = 0.01 * np.random.randn(prev_layer.get_shape()[0], nodes)
        self.s = np.empty((nodes + 1,prev_layer.get_shape()[1]))
        self.d = np.empty((nodes + 1,prev_layer.get_shape()[1]))
        
    def forward_prop(self):
        """ This will push the signals forward"""
        self.x = np.dot((self.w).T,self.prev_layer.get_signal())
        
        examples = self.prev_layer.get_shape()[1]
        bias = np.full((1, examples),1)
                
        self.s = np.concatenate((bias, self.activation_f(self.x)), axis=0)
        
    def set_prev(self, prev_layer):
        self.prev_layer = prev_layer
        self.w = 0.01 * np.random.randn(prev_layer.get_shape()[0], self.nodes)
        self.s = np.empty((self.nodes + 1,prev_layer.get_shape()[1]))
        
    def update_weights(self, rate, reg_term):
        self.w = self.w - rate*np.dot(self.prev_layer.get_signal(), self.d[1:].T) + reg_term
        
    def calc_delta(self, prev_deltas, prev_weights):
        self.d = self.d_activation_f(self.s) * np.dot(prev_weights, prev_deltas)
        return self.d[1:]
        
    def get_signal(self):
        return self.s
    
    def get_shape(self):
        return self.s.shape
    
    def get_weights(self):
        return self.w
    
class output_layer:
    
    def __init__(self, prev_layer, nodes, activation_f):
        self.nodes = nodes
        self.prev_layer = prev_layer
        self.activation_f = activation_f
        
        self.w = 0.01 * np.random.randn(prev_layer.get_shape()[0], nodes)
        self.s = np.empty((nodes,prev_layer.get_shape()[1]))
        self.d = np.empty((nodes,prev_layer.get_shape()[1]))
        
    def forward_prop(self):
        """ This will push the signals forward"""
        self.x = np.dot((self.w).T,self.prev_layer.get_signal())
        self.s = self.activation_f(self.x)
        
    def set_prev(self, prev_layer):
        self.prev_layer = prev_layer
        self.w = 0.01 * np.random.randn(prev_layer.get_shape()[0], self.nodes)
        self.s = np.empty((self.nodes,prev_layer.get_shape()[1]))
        
    def update_weights(self, rate, reg_term):
        self.w = self.w - rate*np.dot(self.prev_layer.get_signal(), self.d.T) + reg_term
        
    def set_delta(self, new_d):
        self.d = new_d
        
    def get_signal(self):
        return self.s
    
    def get_shape(self):
        return self.s.shape
    
    def get_weights(self):
        return self.w