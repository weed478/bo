from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt


class Package2D(NamedTuple):
    size: tuple[int, int]
    target_stop: int
    deadline: None


class Problem2D(NamedTuple):
    n_stops: int
    cargo_size: tuple[int, int]
    packages: list[Package2D]


class Solution2D(NamedTuple):
    cargo: np.ndarray


def try_fit(cargo: np.ndarray, pkg: Package2D) -> tuple[int, int] | None:
    dx, dy = pkg.size
    for x in range(cargo.shape[0] - dx + 1):
        for y in range(cargo.shape[1] - dy + 1):
            if np.all(cargo[x:x+dx, y:y+dy] == 0):
                return x, y
    return None


def generate_solution_candidate(prob: Problem2D) -> Solution2D:
    cargo = np.zeros(prob.cargo_size, dtype=np.int32)
    rng = np.random.default_rng(seed=42)
    for pkg_i in rng.permutation(len(prob.packages)):
        pkg = prob.packages[pkg_i]
        top_left = try_fit(cargo, pkg)
        if top_left is None:
            continue
        x, y = top_left
        dx, dy = pkg.size
        cargo[x:x+dx, y:y+dy] = pkg_i + 1
    return Solution2D(cargo=cargo)


def plot_solution(sol: Solution2D):
    rng = np.random.default_rng(seed=42)
    colors = rng.integers(0, 256, size=(sol.cargo.max() + 1, 3))
    colors[0] = (0, 0, 0)
    plt.imshow(colors[sol.cargo])
    plt.title('Solution (black means no package)')
    plt.show()


if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)
    prob = Problem2D(
        n_stops=5,
        cargo_size=(10, 10),
        packages=[
            Package2D(size=(rng.integers(1, 5), rng.integers(1, 5)), target_stop=rng.integers(0, 5), deadline=None)
            for _ in range(20)
        ],
    )
    sol = generate_solution_candidate(prob)
    print(sol.cargo)
    plot_solution(sol)
