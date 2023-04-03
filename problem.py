from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    "Package2D",
    "Problem2D",
    "Solution2D",
    "plot_solution",
]


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
    problem: Problem2D


def plot_solution(sol: Solution2D):
    rng = np.random.default_rng(seed=42)
    colors = np.random.randint(0, 256, size=(sol.cargo.max() + 1, 3))
    colors[0] = (0, 0, 0)
    plt.imshow(colors[sol.cargo])
    plt.title('Solution (black means no package)')
    plt.show()
