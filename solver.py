import numpy as np
import dask.bag as db
import itertools
from problem import *
from solution_candidates import generate_solution_candidates
from crossing import cross_solutions


# What part of space is empty
def fitness_fn(solution: Solution2D):
    return (solution.cargo == 0).sum() / (
        solution.cargo.shape[0] * solution.cargo.shape[1]
    )


# Mean time to get out a package
# where:
#   time = mass blocking a package at it's target stop
#   blocking = having bigger y (being closer to the door)
def time_fn(solution: Solution2D):
    target_stops_of_packages = np.ndarray(
        (package.target_stop for package in solution.problem.packages)
    )

    target_stops_of_fields = (
        np.choose(solution.cargo - 1, target_stops_of_packages) * solution.cargo != -1
    )

    later_targets_blocking_per_target = np.ndarray(
        # How many fields with y bigger than current have later target stop than t
        np.cumsum(
            # How many fields with current y have larger stop than t
            np.sum(target_stops_of_fields > t, axis=0, keepdims=true)[::-1],
            axis=1,
        )[::-1]
        # For every target stop
        for t in range(solution.problem.n_stops)
    )

    blocking_per_field = np.choose(
        target_stops_of_fields, later_targets_blocking_per_target
    )

    return np.mean(blocking_per_field) / (
        solution.cargo.shape[0] * solution.cargo.shape[1]
    )


def cost_fn(solution: Solution2D):
    return fitness_fn(solution) + time_fn(solution)


if __name__ == "__main__":
    pop_size = 100

    prob = Problem2D(
        n_stops=5,
        cargo_size=(20, 20),
        packages=[
            Package2D(
                size=(np.random.randint(2, 6), np.random.randint(2, 6)),
                target_stop=np.random.randint(1, 6),
                deadline=None,
            )
            for _ in range(100)
        ],
        mutation_chance=0.1,
        max_mutation_size=3,
        swap_mutation_chance=0.4,
    )

    population = list(itertools.islice(generate_solution_candidates(prob), pop_size))

    for i in range(10):
        print(f"Generation {i}")

        cost = np.array(db.from_sequence(population).map(cost_fn).compute())
        print(f"Avg niceness: {np.mean(niceness)}")

        parents_a = [
            population[i]
            for i in np.random.choice(pop_size, pop_size, p=(cost / cost.sum()))
        ]
        parents_b = [
            population[i]
            for i in np.random.choice(pop_size, pop_size, p=(cost / cost.sum()))
        ]

        population = (
            db.from_sequence(zip(parents_a, parents_b))
            .map(lambda x: cross_solutions(*x))
            .compute()
        )
