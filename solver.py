import numpy as np
import dask.bag as db
from problem import *
from problem import plot_cost_chart
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
    N_GENERATIONS=100

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
        mutation_chance=0.2,
        max_mutation_size=3,
        swap_mutation_chance=0.5,
        row_height=5,
        prev_generation_remain=0.4
    )

    population = db.from_sequence(range(pop_size)).map(lambda _: generate_solution_candidate_with_rows(prob)).compute()
    generations_avg_cost=[]
    split_index=int(pop_size*prob.prev_generation_remain)

    for i in range(N_GENERATIONS):
        print(f"Generation {i}")

        fitness = np.array(db.from_sequence(population).map(fitness_fn).compute())
        print(f"Mean fitness: {np.mean(fitness)}")

        generations_avg_cost.append(100/np.mean(fitness))

        parents_ab = [
            population[i]
            for i in np.random.choice(pop_size, 2 * pop_size, p=fitness / fitness.sum())
        ]

        parents_a = parents_ab[::2]
        parents_b = parents_ab[1::2]
        
        next_generation=(
            db.from_sequence(zip(parents_a, parents_b))
            .map(lambda x: cross_solutions_by_rows(*x))
            .compute()
        )

        np.random.shuffle(population)
        next_generation.sort(key=fitness_fn)
        population=population[pop_size-split_index:]+next_generation[split_index:]

    plot_cost_chart(generations_avg_cost,f"N_GENERATIONS = {N_GENERATIONS}, PREV = {prob.prev_generation_remain}, MUTATION_CHANCE = {prob.mutation_chance}")