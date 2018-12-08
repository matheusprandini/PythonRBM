import numpy as np
np.warnings.filterwarnings('ignore')

##### CLASS RBM VERSION 1 ####

#### RBM WITHOUT BIAS AND MOMENTUM ####
#### RBM WITH FIXED LEARNING RATE ####

class RBM:

    def __init__(self, number_visible_layer_neurons=9, number_hidden_layer_neurons=8, learning_rate=0.1):
		
		# Initialize Weights and Learning rate
        self.weights = self.initialize_weights(number_visible_layer_neurons, number_hidden_layer_neurons)
        self.learning_rate = learning_rate
    

    def initialize_weights(self, number_visible_layer_neurons, number_hidden_layer_neurons):
        # Weights initialization -> 2015 by He et al (similar to Xavier initialization)
        W = np.random.randn(number_hidden_layer_neurons, number_visible_layer_neurons)*np.sqrt(2/number_visible_layer_neurons)
        
        return W
		
    def forward_backward_phase(self, input_data):
	
	    # Forward phase (Visible -> Hidden)
        if input_data.shape[1] == self.weights.shape[1]:
            Z = np.dot(self.weights, input_data.T)
		# Reconstruction phase (Hidden -> Visible)
        else:
            Z = np.dot(self.weights.T, input_data.T)
		
		# Logistic sigmoid
        A = self.sigmoid(Z)
		
        return A.T
		
    def update_weights(self, V0, V1):

	    # Compute delta
        delta = self.learning_rate * (V0 - V1)
        self.weights += delta
		
    def training_nn(self, training_data, epochs=1000):
	
        for i in range(epochs):
            V0 = training_data
		
            H0 = self.forward_backward_phase(V0)
			
            V1 = self.forward_backward_phase(H0)
			
            self.update_weights(V0,V1)
			
    def test_nn(self, test_data):
        V0 = test_data
		
        H0 = self.forward_backward_phase(V0)
			
        V1 = self.forward_backward_phase(H0)
		
        print(V1)
	
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
        