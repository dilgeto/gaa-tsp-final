import numpy as np
import tsplib95
import random
import math
import globals

# TODO: TSP
def swap(args):
    try:
        solution_container = globals.current_solution_container

        # Selecciona dos índices diferentes para intercambiar
        for i in range(solution_container["num_cities"]):
            for j in range(i + 1, solution_container["num_cities"]):
                # Genera una nueva ruta con las ciudades intercambiadas
                new_route = solution_container["route"].copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]

                # Calcula la distancia total antes y después del intercambio
                current_distance = calculate_distance(solution_container["route"], solution_container["distance_matrix"])
                new_distance = calculate_distance(new_route, solution_container["distance_matrix"])

                # Si mejora, actualiza la ruta
                if new_distance < current_distance:
                    solution_container["route"] = new_route
        return True
    except Exception as e:
        print(f"EXCEPTION: Could not apply swap ({e}), continuing code...")
        return False

def invert(args):
    try:
        solution_container = globals.current_solution_container

        # Selecciona dos índices diferentes
        for i in range(1, solution_container["num_cities"] - 2):
            for j in range(i + 1, solution_container["num_cities"]):
                # Genera una nueva ruta con las ciudades invertidas entre i y j
                new_route = solution_container["route"].copy()
                new_route[i:j+1] = reversed(new_route[i:j+1])

                # Calcula la distancia total antes y después de la inversión
                current_distance = calculate_distance(solution_container["route"].copy(), solution_container["distance_matrix"])
                new_distance = calculate_distance(new_route, solution_container["distance_matrix"])

                # Si mejora, actualiza la ruta
                if new_distance < current_distance:
                    solution_container["route"] = new_route
        return True
    except Exception as e:
        print(f"EXCEPTION: Could not apply invert ({e}), continuing code...")
        return False

def two_opt(args):
    try:
        solution_container = globals.current_solution_container

        best = solution_container["route"].copy()
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 1):
                for j in range(i + 1, len(best)):
                    if j - i == 1:  # No intercambiar adyacentes
                        continue
                    new_solution = best[:i] + best[i:j][::-1] + best[j:]
                    new_cost = calculate_distance(new_solution, solution_container["distance_matrix"])
                    if new_cost < solution_container["current_distance"]:
                        best = new_solution
                        solution_container["current_distance"] = new_cost
                        improved = True
        solution_container["route"] = best.copy()
        return True
    except Exception as e:
        print(f"EXCEPTION: Could not apply two_opt ({e}), continuing code...")
        return False

### INICIO PARTE DE ANNEALING:
def simulated_annealing(args):
    """Peforms simulated annealing to find a solution"""
    try:
        state = random.getstate()
        random.seed(4683)
        #print("annealing start")
        initial_temp = args["initial_temp"]
        alpha = args["alpha"]

        solution_container = globals.current_solution_container
        distances = solution_container["distance_matrix"]
        
        current_temp = initial_temp

        # Start by initializing the current state with the initial state
        solution = solution_container["route"].copy()
        same_solution = 0
        same_cost_diff = 0
        
        while same_solution < 1500 and same_cost_diff < 150000:
            neighbor = get_neighbors(solution)
            
            # Check if neighbor is best so far
            cost_diff = get_cost(neighbor, distances) - get_cost(solution, distances)
            # if the new solution is better, accept it
            if cost_diff > 0:
                solution = neighbor
                same_solution = 0
                same_cost_diff = 0
                
            elif cost_diff == 0:
                solution = neighbor
                same_solution = 0
                same_cost_diff +=1
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            else:
                if random.uniform(0, 1) <= math.exp(float(cost_diff) / float(current_temp)):
                    solution = neighbor
                    same_solution = 0
                    same_cost_diff = 0
                else:
                    same_solution +=1
                    same_cost_diff+=1
            # decrement the temperature
            current_temp = current_temp*alpha
            #print(1/get_cost(solution), same_solution)
        solution_container["route"] = solution.copy()
        solution_container["current_distance"] = 1/get_cost(solution, distances)
        #print("annealing end")
        random.setstate(state)
        return True
    except Exception as e:
        print(f"EXCEPTION: Could not apply simulated_annealing ({e}), continuing code...")
        return False

def get_cost(state, distances):
    """Calculates cost/fitness for the solution/route."""
    distance = 0.0
    i = 0
    while i < len(state):
        if i == (len(state) - 1):
            distance += distances[state[i]][state[0]]
        else:
            distance += distances[state[i]][state[i+1]]
        i += 1
    fitness = 1/distance
    return fitness

def get_neighbors(state):
    """Returns neighbor of  your solution."""
    #neighbor = copy.deepcopy(state)
    neighbor = state.copy()
        
    
    func = random.choice([0,1,2,3])
    if func == 0:
        inverse(neighbor)
        
    elif func == 1:
        insert(neighbor)
        
    elif func == 2 :
        sawp(neighbor)
    
    else:
        swap_routes(neighbor)
        
    return neighbor 

def inverse(state):
    "Inverses the order of cities in a route between node one and node two"
   
    node_one = random.choice(state)
    new_list = list(filter(lambda city: city != node_one, state)) #route without the selected node one
    node_two = random.choice(new_list)
    state[min(node_one,node_two):max(node_one,node_two)] = state[min(node_one,node_two):max(node_one,node_two)][::-1]
    
    return state

def insert(state):
    "Insert city at node j before node i"
    node_j = random.choice(state)
    state.remove(node_j)
    node_i = random.choice(state)
    index = state.index(node_i)
    state.insert(index, node_j)
    
    return state

def sawp(state):
    "Swap cities at positions i and j with each other"
    pos_one = random.choice(range(len(state)))
    pos_two = random.choice(range(len(state)))
    state[pos_one], state[pos_two] = state[pos_two], state[pos_one]
    
    return state

def swap_routes(state):
    "Select a subroute from a to b and insert it at another position in the route"
    subroute_a = random.choice(range(len(state)))
    subroute_b = random.choice(range(len(state)))
    subroute = state[min(subroute_a,subroute_b):max(subroute_a, subroute_b)]
    del state[min(subroute_a,subroute_b):max(subroute_a, subroute_b)]
    insert_pos = random.choice(range(len(state)))
    for i in subroute:
        state.insert(insert_pos, i)
    return state

### FIN PARTE DE ANNEALING

### INICIO PARTE DE ACO
def ant_colony_optimization(args):
    try:
        state = random.getstate()
        random.seed(4683)
        #print("aco start")
        n_ants = args["n_ants"]
        n_iterations = args["n_iterations"]
        alpha = args["alpha"]
        beta = args["beta"]
        evaporation = args["evaporation"]
        Q = args["Q"]

        solution_container = globals.current_solution_container
        n_cities = solution_container["num_cities"]
        distances = solution_container["distance_matrix"]
        pheromones = initialize_pheromones(n_cities)
        best_path = None
        best_cost = float('inf')
        
        for _ in range(n_iterations):
            paths = []
            costs = []
            
            for _ in range(n_ants):
                path, cost = construct_solution(distances, pheromones, alpha, beta)
                paths.append(path)
                costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
            
            update_pheromones(pheromones, paths, costs, evaporation, Q)
        
        solution_container["route"] = best_path.copy()
        solution_container["current_distance"] = best_cost
        #print("aco end")
        random.setstate(state)
        return True
    except Exception as e:
        print(f"EXCEPTION: Could not apply ant_colony_optimization ({e}), continuing code...")
        return False

def initialize_pheromones(n_cities):
    return np.ones((n_cities, n_cities)) / n_cities

def construct_solution(distances, pheromones, alpha, beta):
    n_cities = len(distances)
    start = random.randint(0, n_cities - 1)
    path = [start]
    visited = set(path)
    
    while len(path) < n_cities:
        current = path[-1]
        next_city = select_next_city(current, visited, distances, pheromones, alpha, beta)
        path.append(next_city)
        visited.add(next_city)
    
    path.append(path[0])  # Hacerlo un ciclo
    cost = calculate_cost(path, distances)
    return path, cost

def select_next_city(current, visited, distances, pheromones, alpha, beta):
    probabilities = []
    total = 0
    
    for j in range(len(distances)):
        if j not in visited:
            pheromone = pheromones[current][j] ** alpha
            heuristic = (1 / distances[current][j]) ** beta if distances[current][j] > 0 else 0
            prob = pheromone * heuristic
            probabilities.append((j, prob))
            total += prob
    
    if total == 0:
        return random.choice([j for j in range(len(distances)) if j not in visited])
    
    r = random.uniform(0, total)
    cumulative = 0
    for city, prob in probabilities:
        cumulative += prob
        if r <= cumulative:
            return city

def calculate_cost(path, distances):
    return sum(distances[path[i]][path[i+1]] for i in range(len(path) - 1))

def update_pheromones(pheromones, paths, costs, evaporation, Q):
    pheromones *= (1 - evaporation)  # Evaporación
    
    for path, cost in zip(paths, costs):
        for i in range(len(path) - 1):
            pheromones[path[i]][path[i+1]] += Q / cost
            pheromones[path[i+1]][path[i]] += Q / cost  # Simetría

### FIN PARTE DE ACO

# AUXILIAR
# TODO: MOVER A UN LUGAR ADECUADO
def calculate_distance(route, distance_matrix):
    i = 0
    cost = 0.0
    while i < len(route):
        if i == (len(route) - 1):
            cost += distance_matrix[route[i]][route[0]]
        else:
            cost += distance_matrix[route[i]][route[i+1]]
        i += 1
    return cost