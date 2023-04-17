import numpy as np
import dask.bag as db
import itertools
from problem import *
from solution_candidates import generate_solution_candidates, generate_solution_candidates_with_rows
from crossing import cross_solutions, cross_solutions_by_rows


def fitness_fn(solution: Solution2D):
    return (solution.cargo == 0).sum() / (solution.cargo.shape[0] * solution.cargo.shape[1])


if __name__ == '__main__':
    pop_size = 10

    prob = Problem2D(
        n_stops=5,
        cargo_size=(36, 50),
        packages=[
            Package2D(size=(np.random.choice((1, 2, 4, 6)), np.random.randint(2, 6)), target_stop=np.random.randint(1, 6), deadline=None)
            for _ in range(300)
        ],
        mutation_chance=0.1,
        max_mutation_size=3,
        swap_mutation_chance=0.4,
        row_height=6,
    )

    # population = list(itertools.islice(generate_solution_candidates(prob), pop_size))
    population = list(itertools.islice(generate_solution_candidates_with_rows(prob), pop_size))

    # print(population[0])
    # plot_solution(population[0])

    for i in range(1):
        print(f'Generation {i}')

        fitness = np.array(db.from_sequence(population).map(fitness_fn).compute())
        print(f'Avg fitness: {np.mean(fitness)}')

        parents_a = [population[i] for i in np.random.choice(pop_size, pop_size, p=fitness / fitness.sum())]
        parents_b = [population[i] for i in np.random.choice(pop_size, pop_size, p=fitness / fitness.sum())]

        cross_solutions_by_rows(parents_a[0], parents_b[1])

        # population = db.from_sequence(zip(parents_a, parents_b)).map(lambda x: cross_solutions_by_rows(*x)).compute()
