from functools import reduce
from operator import add
import random
from neuralnetwork.network import Network


class Optimizer(object):
    def __init__(self, nn_param_choices, config={},
                 random_select=0.1, mutate_chance=0.3):
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.nn_param_choices = nn_param_choices
        self.config = config

    def get_random_network(self):
        network = {}
        for key in self.nn_param_choices:
            network[key] = random.choice(self.nn_param_choices[key])
        return network

    def create_population(self, count, network, random_topology=False):
        population = []

        for _ in range(count):
            network = Network(self.config, self.nn_param_choices)
            if random_topology:
                network.create_random()
            else:
                network.create_set(self.config['network_topology_by_layer'])

            population.append(network)

        return population

    @staticmethod
    def fitness(network):
        return network.accuracy

    def grade(self, pop):
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        children = []
        for _ in range(2):
            child = {}

            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            network = Network(self.config, self.nn_param_choices)
            network.create_set(child)

            # if self.mutate_chance > random.random():
            if True:
                if self.config['mutate_topology']:
                    self.mutate_topology(network)

                if self.config['mutate_weights']:
                    self.mutate_weights(network)

            children.append(network)

        return children

    def mutate_topology(self, network):
        # Choose a random key.
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # Mutate one of the params.
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])

    def mutate_weights(self, network):
        network.mutate_weights()

    def evolve(self, pop):
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.config['retain'])

        # The parents are every network we want to keep.
        parents = graded[:retain_length+1]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
