import numpy as np
import utils_ga.crossover as co
import utils_ga.mutate as mu


def choose_parents(population_mol, population_scores):
    population_scores = [s + 1e-10 for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    parents = np.random.choice(population_mol, p=population_probs, size=2)
    return parents


def reproduce(population_mol, population_scores, mutation_rate):
    parent_a, parent_b = choose_parents(population_mol, population_scores)

    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child
