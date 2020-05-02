import numpy as np

from layers import InputLayer, HiddenLayer, OutputLayer


def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:

	def __init__(self):
		self.input_layer = InputLayer(neurons=2)
		self.hidden_layers = [
			HiddenLayer(neurons=2, connections=2)
			# HiddenLayer(neurons=4, connections=5),
			# HiddenLayer(neurons=3, connections=4)
		]
		self.output_layer = OutputLayer(neurons=1, connections=2)


	def feedforward(self, x):

		self.input_layer = InputLayer(values=x)

		for i, layer in enumerate(self.hidden_layers):
			previous_layer = self.hidden_layers[i-1] if i != 0 else \
				self.input_layer

			for j, neuron in enumerate(layer.neurons):
				neuron.value = np.dot(neuron.weights,
					[p_n.value for p_n in previous_layer.neurons]) + neuron.bias
				layer.neurons[j] = neuron

			self.hidden_layers[i] = layer

		# calculate output layer
		for k, neuron in enumerate(self.output_layer.neurons):
			neuron.value = np.dot(neuron.weights,
				[p_n.value for p_n in self.hidden_layers[-1].neurons]) + neuron.bias
			self.output_layer.neurons[k] = neuron

		return self.output_layer.neurons[0].value

	def calculate_d_y_pred_values(self, sum_o1):

		# calculate d_y_pred_values of output layer
		for i in range(len(self.hidden_layers[-1].neurons)):
			self.output_layer.neurons[0].d_ypred_d_w_values[i] = \
				self.hidden_layers[-1].neurons[i].value \
				* deriv_sigmoid(sum_o1)
			self.hidden_layers[-1].neurons[i].d_ypred_d_h = self.output_layer.neurons[0].weights[i] \
				* deriv_sigmoid(sum_o1)

		# no loop since there is only a single neuron in output
		self.output_layer.neurons[0].d_ypred_d_b = deriv_sigmoid(sum_o1)

		# calculate d_y_pred_values of hidden layers
		for i in range(len(self.hidden_layers)):
			for j in range(len(self.hidden_layers[i].neurons)):
				previous_layer = self.hidden_layers[i-1] if i != 0 else \
					self.input_layer

				for k in range(len(previous_layer.neurons)):
					self.hidden_layers[i].neurons[j].d_ypred_d_w_values[k] = \
						previous_layer.neurons[k].value \
						* deriv_sigmoid(self.hidden_layers[i].neurons[j].value)

				self.hidden_layers[i].neurons[j].d_ypred_d_b = \
					deriv_sigmoid(self.hidden_layers[i].neurons[j].value)

	def update_weights_and_biases(self, d_L_d_ypred, learn_rate):

		for i in range(len(self.output_layer.neurons)):
			for j in range(len(self.output_layer.neurons[i].weights)):
				self.output_layer.neurons[i].weights[j] -= \
					learn_rate * d_L_d_ypred \
					* self.output_layer.neurons[i].d_ypred_d_w_values[j]
			self.output_layer.neurons[i].bias -= \
					learn_rate * d_L_d_ypred \
					* self.output_layer.neurons[i].bias

		for k in range(len(self.hidden_layers)):
			for i in range(len(self.hidden_layers[k].neurons)):
				for j in range(len(self.hidden_layers[k].neurons[i].weights)):
					self.hidden_layers[k].neurons[i].weights[j] -= \
						learn_rate * d_L_d_ypred \
						* self.hidden_layers[k].neurons[i].d_ypred_d_h \
						* self.hidden_layers[k].neurons[i].d_ypred_d_w_values[j]
				self.hidden_layers[k].neurons[i].bias -= \
						learn_rate * d_L_d_ypred \
						* self.hidden_layers[k].neurons[i].d_ypred_d_h \
						* self.hidden_layers[k].neurons[i].bias

	def train(self, data, all_y_trues):
		'''
		- data is a (n x 2) numpy array, n = # of samples in the dataset.
		- all_y_trues is a numpy array with n elements.
		  Elements in all_y_trues correspond to those in data.
		'''
		learn_rate = 0.1
		epochs = 1000 # number of times to loop through the entire dataset

		for epoch in range(epochs):
			for x, y_true in zip(data, all_y_trues):
				self.input_layer = InputLayer(neurons=len(x), values=x)

				sum_o1 = self.feedforward(x)

				y_pred = sigmoid(sum_o1)

				# --- Calculate partial derivatives.
				d_L_d_ypred = -2 * (y_true - y_pred)

				self.calculate_d_y_pred_values(sum_o1)

				# --- Update weights and biases
				self.update_weights_and_biases(d_L_d_ypred, learn_rate)



# # Define dataset
# data = np.array([
#   [-2, -1, 9, -3, 2],  # Alice
#   [25, 6, 2, -8, 2],   # Bob
#   [17, 4, 0, 0, -1],   # Charlie
#   [-15, -6, 6, -2, -8], # Diana
# ])

# Define dataset (height (-135), weight (-66))
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = NeuralNetwork()
network.train(data, all_y_trues)

# Make some predictions
# emily = np.array([-7, -3, 0, 1, 2]) # 128 pounds, 63 inches
# frank = np.array([20, 2, 3, 1, -1])  # 155 pounds, 68 inches
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % sigmoid(network.feedforward(emily))) # 0.951 - F
print("Frank: %.3f" % sigmoid(network.feedforward(frank))) # 0.039 - M
