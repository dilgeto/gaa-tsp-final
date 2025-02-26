import ast
import operator
from functools import partial
from Nodes.nodesMapping import *


def parse_and_execute(tree_str):
    """
    Parses a string representation of a syntactic tree and executes the algorithm it represents.

    Parameters:
    - tree_str: A string representing the syntactic tree (e.g., "OR(AND(low_pass, high_pass), high_pass)").
    - solution_container: The solution container to operate on.

    Returns:
    - The result of executing the syntactic tree.
    """
    def parse_tree(node_str):
        """
        Recursively parse a string into a tree structure using the functions_info dictionary.

        Parameters:
        - node_str: A string representing a part of the syntactic tree.

        Returns:
        - A tuple (function, args), where:
          - function: The callable function for the node.
          - args: A dictionary of arguments for the function.
        """
        # Check if the node is a leaf node (not called like a function)
        if not "(" in node_str:
            return nodes_info_mapping[node_str][0], nodes_info_mapping[node_str][1]

        # Parse the function name and arguments
        tree = ast.parse(node_str, mode='eval').body
        func_name = tree.func.id
        args_list = [ast.unparse(arg) for arg in tree.args]

        # Parse the children recursively
        children = [parse_tree(arg) for arg in args_list]

        # Prepare arguments for the current function
        func = nodes_info_mapping[func_name][0]
        func_args = nodes_info_mapping[func_name][1].copy()

        # Add parsed children as arguments
        for idx, _ in enumerate(children):
            if f"P{idx+1}" in func_args:
                func_args[f"P{idx+1}"] = partial(children[idx][0], children[idx][1])

        return func, func_args

    def execute_node(func, args):
        """
        Executes a single node in the syntactic tree.

        Parameters:
        - func: The function to execute.
        - args: The arguments for the function.

        Returns:
        - The result of the function execution.
        """
        return func(args)

    # Parse the tree from the input string
    root_func, root_args = parse_tree(tree_str)

    # Execute the root of the tree
    return execute_node(root_func, root_args)
