import logging
import os
from neuralnetwork.optimizers.genetic_algorithm import Optimizer
from board.board import Game

# supress tensorflow print messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_race(drivers, num_drivers, time_limit, config):
    return Game(num_cars=num_drivers, config=config).run_with_neural_networks(drivers, time_limit)


def get_average_performance(drivers):
    return sum(driver.accuracy for driver in drivers) / len(drivers)


def set_networks_performance(drivers, cars):
    for driver, car in zip(drivers, cars):
        driver.accuracy = (car.position.x / 1024) * 100


def generate_drivers(nn_param_choices, config={}):
    optimizer = Optimizer(nn_param_choices, config=config)
    # these are the driver networks
    networks = optimizer.create_population(config['num_drivers'], random_topology=config['random_topology'])

    # Evolve the generation.
    for i in range(config['generations']):
        logging.warning("***Doing generation %d of %d***" %
                     (i + 1, config['generations']))

        cars = run_race(networks, config['num_drivers'], config['time_limit'], config)

        check_for_swerving(cars)

        set_networks_performance(networks, cars)

        # this should the be average distance traveled so we can see the
        # improvement from generation to generation
        average_distance = get_average_performance(networks)

        # Print out the average accuracy each generation.
        logging.warning("Generation average: " + str(average_distance) + "%")
        logging.warning('-'*80)

        # Evolve, except on the last iteration.
        if i != config['generations'] - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def check_for_swerving(cars):
    for car in cars:
        for c in range(1,len(car.history)):            
            # going in opposite direction
            if (car.history[c] * car.history[c-1]) < 0:
                print('swerving')

def print_networks(networks):
    logging.warning('-'*80)
    for network in networks:
        network.print_network()


def start_race():
    # nn_param_choices = {
    #     'nb_neurons': [64, 128, 256, 512, 768, 1024],
    #     'nb_layers': [1, 2, 3, 4],
    #     'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
    #     'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
    #                   'adadelta', 'adamax', 'nadam'],
    # }

    nn_param_choices = {
        'nb_neurons': [4],
        'nb_layers': [2],
        'activation': ['tanh'],
        'optimizer': ['sgd'],
    }

    config = {
        'generations': 1,
        'num_drivers': 1,
        'epochs': 1,
        'output_activation': 'tanh',
        'retain': 0.4,
        'input_shape': (5,),
        'nb_classes': 2,
        'time_limit': 20,
        'random_topology': True,
        'mutate_topology': True,
        'mutate_weights': False,
        'sensor_distance': [(70, 70), (90,35), (110, 0), (90, -35), (70, -70)]
    }

    logging.warning("***Evolving %d generations with population %d***" %
                 (config['generations'], config['num_drivers']))

    generate_drivers(nn_param_choices, config)


if __name__ == '__main__':
    start_race()



