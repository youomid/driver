import numpy as np


class Neuron:
	def __init__(self, connections, value=0, bias=np.random.normal()):
		self.weights = [np.random.normal() for _ in range(connections)]
		self.bias = bias
		self.value = value
		self.d_ypred_d_w_values = [0 for _ in range(connections)]
		self.d_y_pred_d_b = 0


class HiddenNeuron(Neuron):
	def __init__(self, connections, value=0, bias=np.random.normal()):
		super().__init__(connections, value, bias)
		self.d_ypred_d_h = 0
