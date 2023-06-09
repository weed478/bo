import numpy as np
import dask.bag as db
from problem import *
from solution_candidates import generate_solution_candidate_with_rows
from crossing import cross_solutions_by_rows
import hydra


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
    return fitness_fill(solution) + solution.problem.alfa * fitness_speed(solution)


@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(conf):
    POP_SIZE = conf.solver.pop_size
    N_GENERATIONS = conf.solver.n_generations

    prob = Problem2D(
        n_stops=conf.problem.n_stops,
        cargo_size=(conf.problem.cargo_size[0], conf.problem.cargo_size[1]),
        packages=[
            Package2D(size=(np.random.choice(conf.problem.package.x_choices), np.random.randint(*conf.problem.package.y_range)),
                      target_stop=np.random.randint(1, conf.problem.n_stops + 1), deadline=None)
            for _ in range(conf.problem.n_packages)
        ],
        mutation_chance=conf.solver.mutation_chance,
        max_mutation_size=conf.solver.max_mutation_size,
        swap_mutation_chance=conf.solver.swap_mutation_chance,
        row_height=conf.solver.row_height,
        alfa=conf.solver.alfa,
    )

    population = db.from_sequence(range(POP_SIZE)).map(lambda _: generate_solution_candidate_with_rows(prob)).compute()
    generations_avg_fitness = []
    generations_best_fitness = []

    for i in range(N_GENERATIONS):
        print(f"Generation {i}")

        fitness = np.array(db.from_sequence(population).map(fitness_fn).compute())

        best_fitness_idx = np.argmax(fitness)
        print(f"Mean fitness: {np.mean(fitness)}")
        print(f"Best fitness: {fitness[best_fitness_idx]}")
        generations_avg_fitness.append(np.mean(fitness))
        generations_best_fitness.append(fitness[best_fitness_idx])
        plot_solution(
            population[best_fitness_idx],
            title=f"Generation {i} best solution (fitness = {fitness[best_fitness_idx]})"
        )

        parents_ab = [
            population[i]
            for i in np.random.choice(POP_SIZE, 2 * POP_SIZE, p=fitness / fitness.sum())
        ]

        parents_a = parents_ab[::2]
        parents_b = parents_ab[1::2]

        children = (
            db.from_sequence(zip(parents_a, parents_b))
            .map(lambda x: cross_solutions_by_rows(*x))
            .compute()
        )

        next_generation = population + children
        next_generation.sort(key=fitness_fn, reverse=True)

        population = next_generation[:len(population)]

    plot_fitness_chart(
        generations_avg_fitness,
        f"POP_SIZE = {POP_SIZE}, "
        f"N_GENERATIONS = {N_GENERATIONS}, "
        f"MUTATION_CHANCE = {prob.mutation_chance}, ",
        "Average solution fitness"
    )

    plot_fitness_chart(
        generations_best_fitness,
        f"POP_SIZE = {POP_SIZE}, "
        f"N_GENERATIONS = {N_GENERATIONS}, "
        f"MUTATION_CHANCE = {prob.mutation_chance}, ",
        "Best solution fitness"
    )


if __name__ == "__main__":
    main()
