import numpy as np
from random import random

def generate_chromosome(n_var, n_bit):
    """
    Generate a chromosome represented as a list of binary values.

    Parameters:
    n_var (int): Number of variables (genes) in the chromosome.
    n_bit (int): Number of bits per variable.

    Returns:
    list: A list representing the chromosome, where each element is either 0 or 1.
    """
    gene_size = n_var * n_bit
    chromosome = []
    for _ in range(gene_size):
        if random() > 0.5:
            chromosome.append(1)
        else:
            chromosome.append(0)
    return chromosome

def generate_gen(n_var, n_bit, ra, rb):
    """
    Generate genes based on the chromosome and map them to a specified range.

    Parameters:
    n_var (int): Number of variables (genes).
    n_bit (int): Number of bits per variable.
    ra (float): The lower bound of the range.
    rb (float): The upper bound of the range.

    Returns:
    tuple: A tuple containing:
        - list: A list of genes, each mapped to the range [ra, rb].
        - list: The chromosome used to generate the genes.
    """
    gen = []
    chromosome = generate_chromosome(n_var, n_bit)
    for i in range(1, n_var+1):
        kr = chromosome[n_bit * (i - 1): n_bit * i]
        for j in range(1, n_bit+1):
            kr[j - 1] = kr[j-1] * (2**(n_bit - j))
        x = np.sum(kr)
        x = x/((2**n_bit) -1)
        gen.append(rb + (ra - rb) * x)
    
    return gen, chromosome