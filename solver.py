import numpy as np
import dask.bag as db
from problem import *
from solution_candidates import generate_solution_candidate_with_rows
from crossing import cross_solutions_by_rows


def fitness_fill(solution: Solution2D):
    """What % of space is occupied"""

    return (solution.cargo != 0).sum() / (solution.cargo.size)


def fitness_speed(solution: Solution2D):
    """Mean speed of getting out a package
    = cargo size / mean mass blocking a package (mass closer to the door)"""

    target_stops_of_packages = np.array(
        [0] + [package.target_stop for package in solution.problem.packages]
    )

    target_stops_of_fields = target_stops_of_packages[solution.cargo]

    # How many fields with y bigger than current
    # have larger target stop than t
    blocking_per_target = [0] + [
        np.cumsum(
            # How many fields with current y have
            # larger stop than t
            np.sum(target_stops_of_fields > t, axis=0)[::-1],
        )[::-1]
        # For every target stop
        for t in range(solution.problem.n_stops)
    ]

    blocking_per_field = target_stops_of_fields.choose(blocking_per_target)

    return solution.cargo.size / np.mean(blocking_per_field)


def fitness_fn(solution: Solution2D):
    return fitness_fill(solution) + fitness_speed(solution)


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
