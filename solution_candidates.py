from typing import Iterator
import numpy as np
from problem import *
from crossing import mutate_solution

def try_fit(cargo: np.ndarray, pkg: Package2D) -> tuple[int, int] | None:
    dx, dy = pkg.size
    for x in range(cargo.shape[0] - dx + 1):
        for y in range(cargo.shape[1] - dy + 1):
            if np.all(cargo[x:x+dx, y:y+dy] == 0):
                return x, y
    return None


def generate_solution_candidates(prob: Problem2D) -> Iterator[Solution2D]:
    while True:
        cargo = np.zeros(prob.cargo_size, dtype=np.int32)
        for pkg_i in np.random.permutation(len(prob.packages)):
            pkg = prob.packages[pkg_i]
            top_left = try_fit(cargo, pkg)
            if top_left is None:
                continue
            x, y = top_left
            dx, dy = pkg.size
            cargo[x:x+dx, y:y+dy] = pkg_i + 1
        yield Solution2D(cargo=cargo, problem=prob)


if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)
    prob = Problem2D(
        n_stops=5,
        cargo_size=(10, 10),
        packages=[
            Package2D(size=(rng.integers(1, 5), rng.integers(1, 5)), target_stop=rng.integers(0, 5), deadline=None)
            for _ in range(100)
        ],
        mutation_chance=0.1,
        max_mutation_size=3,
        swap_mutation_chance=0.4
    )
    sol = generate_solution_candidates(prob)
    print(sol.cargo)
    plot_solution(sol)
