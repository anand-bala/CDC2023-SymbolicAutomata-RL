from typing import Mapping

import networkx as nx
import numpy as np

from syma.automaton import SymbolicAutomaton


def distance_to_accepting_set(sa: SymbolicAutomaton) -> Mapping[int, float]:
    eta = {q: np.inf for q in sa.locations}

    # Get the all pairs shortest path lengths
    lengths = dict(
        nx.all_pairs_shortest_path_length(sa._graph)
    )  # dist[int, dict[int, int]]

    for q, dists in lengths.items():
        if sa[q].accepting:
            eta[q] = 0
            continue
        for q_, d in dists.items():
            if sa[q_].accepting:
                eta[q] = min(eta[q], d)
    return eta
