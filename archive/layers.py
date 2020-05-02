import numpy as np

from neuron import Neuron, HiddenNeuron

class Layer:
	def __init__(self, neurons=1, connections=1):
		# connections represent inbound connections to a neuron
		self.neurons = [Neuron(connections) for _ in range(neurons)]

class InputLayer(Layer):
	def __init__(self, neurons=1, connections=0, values=[]):
		self.neurons = [Neuron(connections, value=value) for value in values]

class HiddenLayer(Layer):
	def __init__(self, neurons=1, connections=1):
		# connections represent inbound connections to a neuron
		self.neurons = [HiddenNeuron(connections) for _ in range(neurons)]	

class OutputLayer(Layer):
	pass





