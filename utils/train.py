from keras.models import Sequential
from keras.layers import Dense, Dropout


def compile_model(network, nb_classes, input_shape, config):
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

    # Output layer.
    model.add(Dense(nb_classes, activation=config['output_activation']))

    return model


def compile_model_by_layer(network, input_shape):
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']

    model = Sequential()

    # Add each layer. Output layer will be last iteration.
    for i in range(nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons[i], activation=activation[i], input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons[i], activation=activation[i]))

    return model

