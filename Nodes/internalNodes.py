def WHILE(args):
    """
    Executes P2 repeatedly as long as P1 returns True, up to a maximum number of iterations.
    Returns the number of iterations executed.

    Parameters:
    - args: Dictionary containing:
        - "P1": Callable that returns a boolean value.
        - "P2": Callable to execute repeatedly.
        - "max_iterations": Maximum number of iterations.

    Returns:
    - int: Number of iterations executed.
    """
    P1 = args["P1"]
    P2 = args["P2"]
    max_iterations = args["max_iterations"]

    iterations = 0
    while P1() and iterations < max_iterations:
        P2()
        iterations += 1
    return iterations


def AND(args):
    """
    Executes P1. If it returns True, executes and returns the result of P2.
    Otherwise, returns False.

    Parameters:
    - args: Dictionary containing:
        - "P1": Callable that returns a boolean value.
        - "P2": Callable to execute if P1 returns True.
        - "solution_container": The solution container to operate on.

    Returns:
    - bool: Result of the AND operation.
    """
    P1 = args["P1"]
    P2 = args["P2"]

    if P1():
        return P2()
    return False


def OR(args):
    """
    Executes P1. If it returns False, executes and returns the result of P2.
    Otherwise, returns True.

    Parameters:
    - args: Dictionary containing:
        - "P1": Callable that returns a boolean value.
        - "P2": Callable to execute if P1 returns False.

    Returns:
    - bool: Result of the OR operation.
    """
    P1 = args["P1"]
    P2 = args["P2"]

    if P1():
        return True
    return P2()


def IF(args):
    """
    Executes P1. If it returns True, executes and returns the result of P2.
    Otherwise, executes and returns the result of P3.

    Parameters:
    - args: Dictionary containing:
        - "P1": Callable that returns a boolean value.
        - "P2": Callable to execute if P1 returns True.
        - "P3": Callable to execute if P1 returns False.

    Returns:
    - Result of P2 if P1 is True, otherwise result of P3.
    """
    P1 = args["P1"]
    P2 = args["P2"]
    P3 = args["P3"]

    if P1():
        return P2()
    return P3()


def NOT(args):
    """
    Executes P1 and returns the negation of the result.

    Parameters:
    - args: Dictionary containing:
        - "P1": Callable that returns a boolean value.
        - "solution_container": The solution container to operate on.

    Returns:
    - bool: Negation of P1's result.
    """
    P1 = args["P1"]

    return not P1()
