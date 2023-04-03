import numpy as np
import dask.bag as db
import itertools
from problem import *
from solution_candidates import generate_solution_candidates
from crossing import cross_solutions


def fitness_fn(solution: Solution2D):
    return (solution.cargo == 0).sum() / (solution.cargo.shape[0] * solution.cargo.shape[1])


if __name__ == '__main__':
    pop_size = 100

    prob = Problem2D(
        n_stops=5,
        cargo_size=(20, 20),
        packages=[
            Package2D(size=(np.random.randint(2, 6), np.random.randint(2, 6)), target_stop=np.random.randint(1, 6), deadline=None)
            for _ in range(100)
        ],
    )

    population = list(itertools.islice(generate_solution_candidates(prob), pop_size))

    for i in range(10):
        print(f'Generation {i}')

        fitness = np.array(db.from_sequence(population).map(fitness_fn).compute())
        print(f'Avg fitness: {np.mean(fitness)}')

        parents_a = [population[i] for i in np.random.choice(pop_size, pop_size, p=fitness / fitness.sum())]
        parents_b = [population[i] for i in np.random.choice(pop_size, pop_size, p=fitness / fitness.sum())]

        population = db.from_sequence(zip(parents_a, parents_b)).map(lambda x: cross_solutions(*x)).compute()
