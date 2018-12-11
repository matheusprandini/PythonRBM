import numpy as np
np.warnings.filterwarnings('ignore')

##### CLASS RBM VERSION 2 ####

#### RBM WITHOUT MOMENTUM ####
#### RBM WITH FIXED LEARNING RATE ####

class RBM:

    def __init__(self, number_visible_layer_neurons=9, number_hidden_layer_neurons=50, learning_rate=0.1):
		
		# Initialize Weights, Bias and Learning rate
        self.parameters = {}
        self.learning_rate = learning_rate
        self.initialize_parameters(number_visible_layer_neurons, number_hidden_layer_neurons)
    

    def initialize_parameters(self, number_visible_layer_neurons, number_hidden_layer_neurons):
	
        # Weights initialization -> 2015 by He et al (similar to Xavier initialization)
        W = np.random.randn(number_hidden_layer_neurons, number_visible_layer_neurons)*np.sqrt(2/number_visible_layer_neurons)
        
		# Bias initialization
        b1 = np.zeros(shape=(number_visible_layer_neurons, 1)) # Visible layer bias
        b2 = np.zeros(shape=(number_hidden_layer_neurons, 1)) # Hidden layer bias
		
        self.parameters['W'] = W
        self.parameters['b1'] = b1
        self.parameters['b2'] = b2
		
    def forward_backward_phase(self, input_data):
	
        # Forward phase (Visible -> Hidden)
        if input_data.shape[1] == self.parameters['W'].shape[1]:
            Z = np.dot(self.parameters['W'], input_data.T) + self.parameters['b2']
	    # Reconstruction phase (Hidden -> Visible)
        else:
            Z = np.dot(self.parameters['W'].T, input_data.T) + self.parameters['b1']
		
		# Logistic tanh
        A = np.tanh(Z)

        return A.T
		
    def update_parameters(self, V0, V1, H0, H1):
	
	    # Deltas for visible and hidden
        delta_visible = (self.learning_rate * (V0-V1))
        delta_hidden = (self.learning_rate * (H0-H1))
		
        self.parameters['W'] = self.parameters['W'] + delta_visible
        self.parameters['b1'] = self.parameters['b1'] + delta_visible.T
        self.parameters['b2'] = self.parameters['b2'] + delta_hidden.T
		
    def training_nn(self, training_data, epochs=1000):
	
        for i in range(epochs):
            for data in training_data:
                V0 = np.resize(data, (1,9))

                H0 = self.forward_backward_phase(V0)
			
                V1 = self.forward_backward_phase(H0)
			
                H1 = self.forward_backward_phase(V1)
			
                self.update_parameters(V0,V1,H0,H1)
			
    def test_nn(self, test_data):
        V0 = test_data
		
        H0 = self.forward_backward_phase(V0)
			
        V1 = self.forward_backward_phase(H0)
		
        #print("Hidden Layer: ", H0)
        print("Reconstructed data: ", V1)
	
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
        