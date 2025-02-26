from Nodes.internalNodes import *
from Nodes.terminalNodes import *

nodes_info_mapping = {
    # Internal Node Functions
    "WHILE": [
        WHILE,  # The function as a callable
        {
            "P1": None,              # Placeholder for the first predicate
            "P2": None,              # Placeholder for the second operation
            "max_iterations": 5    # Default maximum iterations
        }
    ],
    "AND": [
        AND,
        {
            "P1": None,  # Placeholder for the first predicate
            "P2": None   # Placeholder for the second operation
        }
    ],
    "OR": [
        OR,
        {
            "P1": None,  # Placeholder for the first predicate
            "P2": None   # Placeholder for the second operation
        }
    ],
    "IF": [
        IF,
        {
            "P1": None,  # Placeholder for the condition
            "P2": None,  # Placeholder for the operation if true
            "P3": None   # Placeholder for the operation if false
        }
    ],
    "NOT": [
        NOT,
        {
            "P1": None  # Placeholder for the predicate
        }
    ],

    # Leaf Node Functions
    "swap": [
        swap,
        {
            # No additional arguments required
        }
    ],
    "invert": [
        invert,
        {
            # No additional arguments required
        }
    ],
    "two_opt": [
        two_opt,
        {
            # No additional arguments required
        }
    ],
    "simulated_annealing": [
        simulated_annealing,
        {
            "initial_temp": 1000,
            "alpha": 0.99
        }
    ],
    "ant_colony_optimization": [
        ant_colony_optimization,
        {
            "alpha": 1,
            "beta": 2,
            "evaporation": 0.5,
            "Q": 100,
            "n_ants": 30,
            "n_iterations": 100
        }
    ],
}