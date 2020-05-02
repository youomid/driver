from keras.models import Sequential
from keras.layers import Dense, Dropout


def compile_model(network, nb_classes, input_shape, config, weights=[]):
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # # Add each layer.
    # for i in range(nb_layers):

    #     # Need input shape for first layer.
    #     if i == 0:
    #         model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
    #     else:
    #         model.add(Dense(nb_neurons, activation=activation))

    #     model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation=config['output_activation']))

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    if weights:
        model.set_weights(weights)

    return model
