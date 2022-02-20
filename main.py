import logging
import os
import csv

from neuralnetwork.optimizers.genetic_algorithm import EvolutionaryOptimizer
from board.board import Game

# suppress tensorflow print messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GAME_WINDOW_HEIGHT = 768
GAME_WINDOW_WIDTH = 1024


def generate_game(drivers, num_drivers, time_limit, config):
    return Game(height=GAME_WINDOW_HEIGHT, width=GAME_WINDOW_WIDTH, num_cars=num_drivers, config=config, track=config['track']).run_with_neural_networks(drivers, time_limit)


def run_race(drivers, num_drivers, time_limit, config):
    return generate_game(drivers, num_drivers, time_limit, config)


def get_average_performance(drivers):
    return sum(driver.accuracy for driver in drivers) / len(drivers)


def set_drivers_performance(drivers, cars):
    # TODO: update this to work with different tracks
    for driver, car in zip(drivers, cars):
        driver.accuracy = (car.position.x / GAME_WINDOW_WIDTH) * 100


def save_car_history(generation, cars):
    for i, car in enumerate(cars):
        column_names = car.history[0].keys()
        with open(f'data/history/gen{generation}_car{i}.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, column_names)
            dict_writer.writeheader()
            dict_writer.writerows(car.history)


def print_car_stats(cars):
    for i, car in enumerate(cars):
        print(i, car.stats)


def generate_drivers(nn_param_choices, config={}):
    optimizer = EvolutionaryOptimizer(nn_param_choices, config=config)
    # these are the driver networks
    drivers = optimizer.create_drivers(config['num_drivers'], config['network_topology_by_layer'],
                                           random_topology=config['random_topology'])

    # Evolve the generation.
    for i in range(config['generations']):
        logging.warning("***Generation %d of %d***" %
                        (i + 1, config['generations']))

        cars = run_race(drivers, config['num_drivers'], config['time_limit'], config)

        save_car_history(i, cars)

        set_drivers_performance(drivers, cars)

        # this should the be average distance traveled so we can see the
        # improvement from generation to generation
        average_distance = get_average_performance(drivers)

        print_car_stats(cars)

        # Print out the average accuracy each generation.
        logging.warning("Generation average: " + str(average_distance) + "%")
        logging.warning('-' * 80)

        # Evolve, except on the last iteration.
        if i != config['generations'] - 1:
            # Do the evolution.
            drivers = optimizer.evolve(drivers)

    # Sort our final population by performance
    drivers = sorted(drivers, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 drivers.
    print_drivers(drivers[:5])


def print_drivers(networks):
    logging.warning('-' * 80)
    for network in networks:
        network.print_network()


def start_race():
    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
    }

    config = {
        'generations': 10,
        'num_drivers': 5,
        'epochs': 10,
        'output_activation': 'tanh',
        'retain': 0.4,
        'input_shape': (5,),
        'nb_classes': 2,
        'time_limit': 10,
        'random_topology': False,
        'mutate_topology': False,
        'mutate_weights': True,
        'sensor_distances': [(70, 70), (90, 35), (110, 0), (90, -35), (70, -70)],
        'track': 'SPIRAL',  # SPIRAL, RANDOM, HORIZONTAL
        'network_topology_by_layer': {
            'nb_neurons': [5, 5, 1],
            'nb_layers': 3,
            'activation': ['relu', 'relu', 'softmax'],
        }
    }

    logging.warning("***Evolving %d generations with population %d***" %
                    (config['generations'], config['num_drivers']))

    generate_drivers(nn_param_choices, config)


if __name__ == '__main__':
    start_race()
