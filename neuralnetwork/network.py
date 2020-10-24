import random
import logging
import numpy as np
from utils.train import compile_model, compile_model_by_layer


class Network(object):
    def __init__(self, config={}, nn_param_choices=None):
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}
        self.model = None
        self.config = config
        self.nb_classes = config['nb_classes']
        self.input_shape = config['input_shape']

    def create_random(self):
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])
        self.create_model()

    def create_model(self):
        if self.config['random_topology']:
            self.model = compile_model(self.network, self.nb_classes, self.input_shape, self.config)
        else:
            self.model = compile_model_by_layer(self.network, self.input_shape)
        self.set_weights()

    def create_set(self, network):
        self.network = network
        self.create_model()

    def drive(self, sensors):
        output = self.model.predict(np.array([np.array(sensors)]))

        # set second return value to range 0 through 1
        return output[0][0], (output[0][1] + 1.0) / 2.0

    def print_network(self):
        logging.warning(self.network)
        logging.warning("Network accuracy: %.2f%%" % (self.accuracy))

    @staticmethod
    def get_mutation_amount():
        return random.choice([-100, 100]) * random.random()

    def set_weights(self):
        weights = self.model.get_weights()

        for i in range(len(weights)):
            if isinstance(weights[i], np.ndarray):
                for j in range(len(weights[i])):
                    if isinstance(weights[i][j], np.ndarray):
                        for k in range(len(weights[i][j])):
                            weights[i][j][k] = random.choice([-2, 2]) * random.random()

        self.model.set_weights(weights)

    def mutate_weights(self):
        weights = self.model.get_weights()

        for i in range(len(weights)):
            if isinstance(weights[i], np.ndarray):
                for j in range(len(weights[i])):
                    if isinstance(weights[i][j], np.ndarray):
                        for k in range(len(weights[i][j])):
                            weights[i][j][k] = self.get_mutation_amount()

        self.model.set_weights(weights)
