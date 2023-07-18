from math import sqrt

def quadratic_roots(a, b, c):
    """
    Quadratic equation root finding
    :param a: The coefficient of the x^2 term
    :param b: The coefficient of the x term
    :param c: The coefficient of the constant term
    """
    
    disc = b ** 2 - 4 * a * c
    if disc < 0: raise ArithmeticError
    elif disc == 0:
        return [(-b + sqrt(disc)) / (2 * a)]

    return [(-b + sqrt(disc)) / (2 * a), (-b - sqrt(disc)) / (2 * a)]