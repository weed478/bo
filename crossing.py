import numpy as np
from problem import *


def try_fit_tl(cargo: np.ndarray, pkg: Package2D) -> tuple[int, int] | None:
    dx, dy = pkg.size
    for x in range(cargo.shape[0] - dx + 1):
        for y in range(cargo.shape[1] - dy + 1):
            if np.all(cargo[x:x+dx, y:y+dy] == 0):
                return x, y
    return None


def try_fit_br(cargo: np.ndarray, pkg: Package2D) -> tuple[int, int] | None:
    dx, dy = pkg.size
    for x in reversed(range(cargo.shape[0] - dx + 1)):
        for y in reversed(range(cargo.shape[1] - dy + 1)):
            if np.all(cargo[x:x+dx, y:y+dy] == 0):
                return x, y
    return None


def cross_solutions(s1: Solution2D, s2: Solution2D) -> Solution2D:
    prob = s1.problem

    s1_pkgs = np.unique(s1.cargo)
    s2_pkgs = np.unique(s2.cargo)
    s1_pkgs = s1_pkgs[s1_pkgs != 0] - 1
    s2_pkgs = s2_pkgs[s2_pkgs != 0] - 1
    s1_pkgs = np.random.permutation(s1_pkgs)
    s2_pkgs = np.random.permutation(s2_pkgs)

    cargo = np.zeros_like(s1.cargo)
    for pkg1_i, pkg2_1 in zip(s1_pkgs, s2_pkgs):
        for pkg_i, try_fit in ((pkg1_i, try_fit_tl), (pkg2_1, try_fit_br)):
            pkg = prob.packages[pkg_i]
            top_left = try_fit(cargo, pkg)
            if top_left is None:
                continue
            x, y = top_left
            dx, dy = pkg.size
            cargo[x:x+dx, y:y+dy] = pkg_i + 1
    return Solution2D(cargo=cargo, problem=prob)
