import numpy as np



class input_layer: 
        
    def __init__(self, inX):
        ''' 
        This intializes an input layer
        
        Args:
            inX: input data with any amount of examples
        '''
        examples = inX.shape[0]
        bias = np.full((examples, 1),1)
        self.x = (np.concatenate((bias, inX), axis=1)).T
        self.s = self.x
        
    def set_input(self,inX):
        '''
        This function will change the inputs, useful for SGD, and passing in test data
        
        Args:
            inX: input data with any amount of examples
        '''
        examples = inX.shape[0]
        bias = np.full((examples, 1),1)
        self.x = (np.concatenate((bias, inX), axis=1)).T
        self.s = self.x

    def get_signal(self):
        '''
        Signal output by this layer
        
        Returns:
            signal of input data, contains the bias node
        '''
        return self.s
    
    def get_shape(self):
        '''
        Convenience function to return the shape of the layer
        
        Returns:
            shape of signal
        '''
        return self.s.shape

    
class hidden_layer:
    
    def __init__(self, prev_layer, nodes, activation_f, d_activation_f):
        '''
        Initializing a hidden layer. This is the main part of your network
        
        Args:
            prev_layer: A reference to the layer before
            nodes: amount of nodes(not including bias)
            activation_f: callable function that accepts a numpy array
            d_activation_f: callable function for the derivative of the activation_f
        '''
        self.nodes = nodes
        self.prev_layer = prev_layer
        self.activation_f = activation_f
        self.d_activation_f = d_activation_f
        
        self.w = 0.01 * np.random.randn(prev_layer.get_shape()[0], nodes)
        self.s = np.empty((nodes + 1,prev_layer.get_shape()[1]))
        self.d = np.empty((nodes + 1,prev_layer.get_shape()[1]))
        
    def forward_prop(self):
        """
        This will push the signals forward
        """
        self.x = np.dot((self.w).T,self.prev_layer.get_signal()) # Wieghts times the previous signal
        
        examples = self.prev_layer.get_shape()[1]
        bias = np.full((1, examples),1) # Generating the bias
                
        self.s = np.concatenate((bias, self.activation_f(self.x)), axis=0)
        
    def set_prev(self, prev_layer):
        """
        In order to change the architecture of the network, the previous layer may need to change. Therefor we will need a new weight array
        
        Args:
            prev_layer: Reference to previous layer
        """
        self.prev_layer = prev_layer
        self.w = 0.01 * np.random.randn(prev_layer.get_shape()[0], self.nodes)
        self.s = np.empty((self.nodes + 1,prev_layer.get_shape()[1]))
        
    def update_weights(self, rate, reg_term):
        """
        When we want to update the weights(After backprop is complete), this will do it
        
        Args:
            rate: Learning rate to apply
            reg_term: the update due to the regularization function
        """
        self.w = self.w - rate*np.dot(self.prev_layer.get_signal(), self.d[1:].T) + reg_term
        
    def reinit_weights(self):
        """
        Used to create new weights, without changing the previous layer
        """
        self.w = 0.01 * np.random.randn(self.prev_layer.get_shape()[0], self.nodes)
        
    def calc_delta(self, prev_deltas, prev_weights):
        """
        Calculating the gradient during backpropagation
        
        Returns:
            Gradients of this later, without the bias gradient
        """
        self.d = self.d_activation_f(self.s) * np.dot(prev_weights, prev_deltas)
        return self.d[1:]
        
    def get_signal(self):
        """
        To get the signal
        
        Returns:
            Layers' current signal
        """
        return self.s
    
    def get_shape(self):
        '''
        Convenience function to return the shape of the layer
        
        Returns:
            shape of signal
        '''
        return self.s.shape
    
    def get_weights(self):
        '''
        Gets wieghts of layer
        
        Returns:
            Layers' weights
        '''
        return self.w
    
class output_layer:
    
    def __init__(self, prev_layer, nodes, activation_f):
        '''
        Initializing a hidden layer. This is the main part of your network
        
        Args:
            prev_layer: A reference to the layer before
            nodes: amount of nodes(not including bias)
            activation_f: callable function that accepts a numpy array
        '''
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
        """
        In order to change the architecture of the network, the previous layer may need to change. Therefor we will need a new weight array
        
        Args:
            prev_layer: Reference to previous layer
        """
        self.prev_layer = prev_layer
        self.w = 0.01 * np.random.randn(prev_layer.get_shape()[0], self.nodes)
        self.s = np.empty((self.nodes,prev_layer.get_shape()[1]))
        
    def update_weights(self, rate, reg_term):
        """
        When we want to update the weights(After backprop is complete), this will do it
        
        Args:
            rate: Learning rate to apply
            reg_term: the update due to the regularization function
        """
        self.w = self.w - rate*np.dot(self.prev_layer.get_signal(), self.d.T) + reg_term
        
    def reinit_weights(self):
        """
        Used to create new weights, without changing the previous layer
        """
        self.w = 0.01 * np.random.randn(self.prev_layer.get_shape()[0], self.nodes)
        
    def set_delta(self, new_d):
        """
        Sets the gradient during backpropagation. We calclate this at the network level
        
        Args:
            new_d: Numpy array of new gradients
        """
        self.d = new_d
        
    def get_signal(self):
        """
        To get the signal
        
        Returns:
            Layers' current signal
        """
        return self.s
    
    def get_shape(self):
        '''
        Convenience function to return the shape of the layer
        
        Returns:
            shape of signal
        '''
        return self.s.shape
    
    def get_weights(self):
        '''
        Gets wieghts of layer
        
        Returns:
            Layers' weights
        '''
        return self.w