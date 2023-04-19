import numpy as np
import dask.bag as db
from problem import *
from solution_candidates import generate_solution_candidate_with_rows
from crossing import cross_solutions_by_rows


# What % of space is occupied by packages
def fitness_fill(solution: Solution2D):
    return (solution.cargo != 0).sum() / (
        solution.cargo.shape[0] * solution.cargo.shape[1]
    )


# Mean time to get out a package
# where:
#   time = mass blocking a package at it's target stop
#   blocking = having bigger y (being closer to the door)
def fitness_time(solution: Solution2D):
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
            np.sum(target_stops_of_fields > t, axis=0, keepdims=True)[::-1],
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


def fitness_fn(solution: Solution2D):
    return fitness_fill(solution)  # + fitness_time(solution)


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
        row_height=5,
    )

    population = db.from_sequence(range(pop_size)).map(lambda _: generate_solution_candidate_with_rows(prob)).compute()

    for i in range(10):
        print(f"Generation {i}")

        fitness = np.array(db.from_sequence(population).map(fitness_fn).compute())
        print(f"Mean fitness: {np.mean(fitness)}")

        parents_ab = [
            population[i]
            for i in np.random.choice(pop_size, 2 * pop_size, p=fitness / fitness.sum())
        ]

        parents_a = parents_ab[::2]
        parents_b = parents_ab[1::2]

        population = (
            db.from_sequence(zip(parents_a, parents_b))
            .map(lambda x: cross_solutions_by_rows(*x))
            .compute()
        )
