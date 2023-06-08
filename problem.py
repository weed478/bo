from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    "Package2D",
    "Problem2D",
    "Solution2D",
    "plot_solution",
    "plot_fitness_chart",
    "plot_cargo"
]


class Package2D(NamedTuple):
    size: tuple[int, int]
    target_stop: int
    deadline: None


class Problem2D(NamedTuple):
    n_stops: int
    cargo_size: tuple[int, int]
    packages: list[Package2D]
    mutation_chance: float
    max_mutation_size: int
    swap_mutation_chance: float
    row_height: int
    alfa: float


class Solution2D(NamedTuple):
    cargo: np.ndarray
    problem: Problem2D


def plot_solution(sol: Solution2D, title='Solution (black square means no package)'):
    rng = np.random.default_rng(seed=42)
    colors = np.random.randint(0, 256, size=(sol.cargo.max() + 1, 3))
    colors[0] = (0, 0, 0)
    plt.figure(figsize=(16, 9))
    plt.imshow(colors[sol.cargo])
    for y in range(sol.cargo.shape[0]):
        for x in range(sol.cargo.shape[1]):
            stop = sol.problem.packages[sol.cargo[y, x] - 1].target_stop
            plt.text(x - 0.25, y + 0.25, str(stop), fontsize=8)
    plt.title(title)
    plt.show()


def plot_fitness_chart(generations_fitness: list, title: str, y_label: str):
    plt.plot(range(len(generations_fitness)), generations_fitness)
    plt.xlabel("Generation #")
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_cargo(cargo: np.ndarray, title='Row'):
    colors = np.random.randint(0, 256, size=(cargo.max() + 1, 3))
    colors[0] = (0, 0, 0)
    plt.imshow(colors[cargo])
    plt.title(title)
    plt.show()
