from nn_layers import input_layer, hidden_layer, output_layer

class nn:
    
    def __init__(self, inputData, output_nodes, output_function):
        inputl = input_layer(inputData)
        outputl = output_layer(inputl,output_nodes,output_function)
        self.layers_list = [inputl,outputl]
        
    def append_hidden_layer(self, layer_nodes, layer_function):
        prevl = self.layers_list[-2]
        hiddenl = hidden_layer(prevl, layer_nodes, layer_function)
        self.layers_list.insert(-1,hiddenl)
        
    def forward_prop(self):
        for i in self.layers_list[1:]:
            i.forward_prop()
            
    def calc_output(self):
        self.forward_prop()
        return self.layers_list[-1].get_signal()
    
    def print_signals(self):
        for i in self.layers_list:
            print(i.get_signal())