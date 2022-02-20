# Driver AI using evolutionary genetic algorithm with neural networks

This goal of this project is to create a race simulation using pyGame and neural networks to see how drivers improve with each race using an evolutionary genetic algorithm. The top neural networks are taken from each race and paired together to create 'children' where the neural network configuration is based on the 'parents' with the possibility of mutations in the neural network's weights. If there is room for more drivers, the optimizer will introduce new random neural networks.

This simulation can be configured in main.py in the start_race function.

![Alt text](/image/simulation.png?raw=true "simulation")
