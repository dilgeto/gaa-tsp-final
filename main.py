#import tensorflow as tf
import numpy
import time
import tsplib95
import math
import random
from deap import base, creator, gp, tools, algorithms

import globals
from Nodes.parseAndExecute import parse_and_execute

def read_instance(nombre_archivo):
    ciudades_raw = tsplib95.load(nombre_archivo + '.tsp')
    matriz = []
    for ciudad_inicio in ciudades_raw.get_nodes():
        x1, y1 = ciudades_raw.node_coords[ciudad_inicio]
        fila_ciudad = []
        for ciudad_destino in ciudades_raw.get_nodes():
            x2, y2 = ciudades_raw.node_coords[ciudad_destino]
            coste = math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))
            fila_ciudad.append(coste)
        matriz.append(fila_ciudad)
    return matriz

def greedy_algorithm(num_cities, distance_matrix):
    """Heurística que genera una solución inicial usando el vecino más cercano."""
    unvisited = set(range(num_cities))
    solution = []
    current = list(unvisited)[0]
    while unvisited:
        unvisited.remove(current)
        solution.append(current)
        if unvisited:
            current = min(unvisited, key=lambda x: distance_matrix[current][x])
    return solution

def calculate_distance(distance_matrix, route):
    i = 0
    cost = 0.0
    while i < len(route):
        if i == (len(route) - 1):
            cost += distance_matrix[route[i]][route[0]]
        else:
            cost += distance_matrix[route[i]][route[i+1]]
        i += 1
    return cost

def read_optimum_instance(nombre_archivo, distancias):
    with open(nombre_archivo + '.opt.tour', 'r') as f:
        tour = [(int(line.strip())-1) for line in f.readlines() if line.strip().isdigit()]
    return calculate_distance(distancias, tour)

check = True

def placheolder():
    return

def create_solution_container(instance, num_cities):
    """
    Create a solution container for the MNIST image.

    Parameters:
    - image: 2D array representing the 28x28 grayscale image.
    - label: True label of the image.

    Returns:
    - solution_container: Dictionary containing the feature vector, class scores,
                          true label, predicted label, and accuracy.
    """
    # Step 1: Generate the distances matrix
    distance_matrix = read_instance(instance)
    data = tsplib95.load(instance + '.tsp')

    # Step 2: Apply greedy to an instance
    greedy_route = greedy_algorithm(num_cities, distance_matrix)

    # Step 3: Calculate the total distance of greedy
    current_distance = calculate_distance(distance_matrix, greedy_route)

    # Step 4: Get the optimum distance
    optimum_distance = read_optimum_instance(instance, distance_matrix)

    # Step 5: Initialize the solution container
    solution_container = {
        "num_cities": num_cities,
        "distance_matrix": distance_matrix,
        "current_distance": current_distance,
        "route": greedy_route[:],
        "optimum_distance": optimum_distance,
        "data": data,
    }

    return solution_container

def obtain_tsp_data():
    # Load the TSPLIB dataset
    # Creating dataset
    solution_container_list = []
    solution_container_list.append(create_solution_container("berlin52",52))
    solution_container_list.append(create_solution_container("ch150",150))
    return solution_container_list

# Set up DEAP's GP system
def setup_gp(dataset):
    """
    Set up the Genetic Programming framework using DEAP.
    Returns the toolbox and other necessary components.
    """
    # Create PrimitiveSet
    pset = gp.PrimitiveSetTyped("MAIN", [], str, "solution")

    # Add Internal Nodes as Primitives
    pset.addPrimitive(placheolder, [str, str], str, "WHILE")
    pset.addPrimitive(placheolder, [str, str], str, "AND")
    pset.addPrimitive(placheolder, [str, str], str, "OR")
    pset.addPrimitive(placheolder, [str, str, str], str, "IF")
    pset.addPrimitive(placheolder, [str], str, "NOT")

    # Add Leaf Nodes as Terminals
    pset.addTerminal(placheolder, str, "swap")
    pset.addTerminal(placheolder, str, "invert")
    pset.addTerminal(placheolder, str, "two_opt")
    pset.addTerminal(placheolder, str, "simulated_annealing")
    pset.addTerminal(placheolder, str, "ant_colony_optimization")

    # Define the fitness function and individual
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))  # Maximize accuracy
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Create the toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness Function
    def evaluate_algorithm(data, individual):
        """
        Evaluate the fitness of an individual syntactic tree on the solution container.

        Parameters:
        - individual: The syntactic tree (algorithm).

        Returns:
        - Tuple containing the fitness (accuracy or other metric).
        """
        # TODO: YO
        # Transform the tree into a callable function
        #global check
        #if check:
        #    print(individual)
        #    check = False
        fitness = 0.0
        num_instances = len(data)

        # For every solution container created from the city sample
        for solution_container in data:
            globals.current_solution_container = solution_container.copy()

            # Apply the generated algorithm
            try:
                parse_and_execute(str(individual))
            except Exception as e:
                print(f"Error evaluating individual: {e}")
                return (float('-inf'),)  # Return worst fitness in case of failure

            # Evaluate the accuracy (example metric)
            fitness += abs(globals.current_solution_container["current_distance"] - globals.current_solution_container["optimum_distance"])/globals.current_solution_container["optimum_distance"]


        algorithm_error = fitness / num_instances

        return (-algorithm_error,)  # Fitness is accuracy

    # Register evaluation function
    toolbox.register("evaluate", evaluate_algorithm, dataset)

    # Register genetic operators
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("compile", gp.compile, pset=pset)

    return toolbox, pset, evaluate_algorithm


def main():
    random.seed(9000)
    """
    Main function to run the Genetic Programming framework.
    """
    # Obtaining MNIST data
    dataset = obtain_tsp_data()

    # Setup GP
    toolbox, pset, eval_func = setup_gp(dataset)

    # Generate initial population
    population = toolbox.population(n=20)

    # Register statistics and Hall of Fame
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Run Genetic Programming
    start = time.time()
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.16, ngen=10, verbose=True, stats=stats,
                                       halloffame=hof)
    end = time.time()

    best_indiv = hof[0]
    print("Best Individual: ", best_indiv)
    print("SCORE: ", eval_func(dataset, best_indiv)[0])
    print(f" --- Algorithm Time: {end-start} --- ")


if __name__ == "__main__":
    main()
