from RBMVersion2 import RBM
import numpy as np

rbm = RBM()

input_data = np.array([[1,-1,-1,-1,1,-1,-1,-1,1],[-1,-1,1,-1,1,-1,1,-1,-1]])

rbm.training_nn(input_data)

rbm.test_nn(input_data)