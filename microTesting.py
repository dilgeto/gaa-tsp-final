import globals
from Nodes.parseAndExecute import parse_and_execute


def evaluate_algorithm(name):
    score = 0
    splitten = globals.test_container[0].split(" ")
    ideal = globals.ideal_test_container.split(" ")
    i = 0
    while i < 6:
        if splitten[i].lower() == ideal[i].lower():
            score += 1
        if splitten[i] == splitten[i].capitalize():
            score += 0.1
        i += 1

    return score  # Fitness is accuracy

treeStr = "IF(NOT(OR(low_pass, dot_product)), WHILE(region_pooling, rank_based_scoring), IF(AND(dot_product, region_pooling), high_pass, dot_product))"
print(globals.test_container)
parse_and_execute(treeStr)
print("Score: ", evaluate_algorithm(globals.test_container[0]))
print(globals.test_container)